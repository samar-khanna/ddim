# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block, Mlp

from models.pos_embed import get_2d_sincos_pos_embed
from models.time_embed import timestep_embedding, ZerosLike


class DDPMaskedAutoencoder(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 temb_dim=0, dropout=0.1,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mask_ratio=0.75,
                 use_add_skip=False, skip_idxs={2: 10, 5: 7, 8: 4, 11: 1}, use_final_conv=True,):
        super().__init__()

        self.in_c = in_chans

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        self.mask_ratio = mask_ratio

        # Temporal embedding stuff for diffusion
        self.temb = nn.Identity()
        self.temb_blocks = nn.ModuleList([nn.Identity()] + [ZerosLike() for _ in range(1, depth)])
        if temb_dim > 0:
            self.temb = nn.Sequential(
                nn.Linear(embed_dim, temb_dim),
                nn.SiLU(),
                nn.Linear(temb_dim, temb_dim),
            )
            self.temb_blocks = nn.ModuleList([
                nn.Sequential(nn.SiLU(), nn.Linear(temb_dim, embed_dim))
                for _ in range(depth)])

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                  norm_layer=norm_layer, drop=dropout)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        self.decoder_temb = nn.Identity()
        self.decoder_temb_blocks = nn.ModuleList([nn.Identity()] + [ZerosLike() for _ in range(1, decoder_depth)])
        if temb_dim > 0:
            self.decoder_temb_blocks = nn.ModuleList([
                nn.Sequential(nn.SiLU(), nn.Linear(temb_dim, decoder_embed_dim))
                for _ in range(decoder_depth)])

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                  norm_layer=norm_layer, drop=dropout)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # Add a projection for each skip connection
        self.use_add_skip = use_add_skip
        self.skip_idxs = skip_idxs
        skip_projs = {}
        for enc_idx, dec_idx in skip_idxs.items():
            skip_projs[dec_idx] = Mlp(
                in_features=embed_dim + decoder_embed_dim, out_features=decoder_embed_dim, drop=dropout
            ) if not use_add_skip else nn.Identity()
        self.skip_projs = nn.ModuleDict({str(i): module for i, module in skip_projs.items()})
        print(f"Using skip connections: {self.skip_idxs}")

        self.final_conv = nn.Conv2d(self.in_c, self.in_c, kernel_size=3, stride=1, padding='same') \
            if use_final_conv else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, patch_size**2 *C)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, p, c):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, temb):
        # embed patches
        x = self.patch_embed(x)  # (N, L, D)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]  # (N, L, D)

        # # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # (N, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, L+1, D)

        # apply Transformer blocks with time
        prev_xs = {}
        for i, blk in enumerate(self.blocks):
            x = blk(x + self.temb_blocks[i](temb))  # (N, L, D)
            if i in self.skip_idxs:
                dec_idx = self.skip_idxs[i]
                prev_xs[dec_idx] = x

        x = self.norm(x)

        # return x, mask, ids_restore
        return x, prev_xs, mask, ids_restore

    def forward_decoder(self, x, prev_xs, temb, mask, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)  # (N, L, D')

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed  # (N, L, D')

        # apply Transformer blocks
        for i, blk in enumerate(self.decoder_blocks):
            if i in prev_xs:
                # This is for deeper layers to use previous xs
                prev_x = prev_xs[i]
                if self.use_add_skip:
                    zeros = torch.zeros(
                        x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], x.shape[2], device=x.device, dtype=x.dtype
                    )
                    prev_x_ = torch.cat([prev_x[:, 1:, :], zeros], dim=1)  # no cls token
                    prev_x_ = torch.gather(prev_x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
                    prev_x = torch.cat([prev_x[:, :1, :], prev_x_], dim=1)  # append cls token

                    x = x + prev_x
                else:
                    raise NotImplementedError("Nah bro.")
                    x = torch.cat((x, prev_x), dim=-1)  # (N, L+1, 2D_t)

                x = self.skip_projs[str(i)](x)  # (N, L+1, D)

            x = blk(x + self.decoder_temb_blocks[i](temb))  # (N, L, D_t)

        x = self.decoder_norm(x)  # (N, L, D_t)

        # predictor projection
        x = self.decoder_pred(x)  # (N, L, P*P*C)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, imgs, time, mask_ratio=0.):
        # embed time
        enc_dim = self.patch_embed.proj.out_channels
        time = timestep_embedding(time, enc_dim)  # (N, D)
        time = time.unsqueeze(1)  # (N, 1, D)
        temb = self.temb(time)  # (N, 1, D_t)

        latent, prev_xs, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, temb)
        pred = self.forward_decoder(latent, prev_xs, temb, mask, ids_restore)  # [N, L, p*p*3]
        mask = mask.unsqueeze(-1).expand_as(pred)  # (N, L, p*p*3)

        pred_img = self.unpatchify(pred, self.patch_embed.patch_size[0], self.in_c)  # (N, C, H, W)
        pred_img = self.final_conv(pred_img)  # (N, C, H, W)

        mask = self.unpatchify(mask, self.patch_embed.patch_size[0], self.in_c)  # (N, C, H, W)
        return pred_img
