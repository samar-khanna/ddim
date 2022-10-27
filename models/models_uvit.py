# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer as vit

from models.pos_embed import get_2d_sincos_pos_embed
from models.time_embed import get_timestep_embedding, ZerosLike


class UVisionTransformer(vit.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, temb_dim=0, skip_rate=1, use_add_skip=False, use_final_conv=True, **kwargs):
        kwargs['qkv_bias'] = True
        super().__init__(**kwargs)

        self.patch_size = kwargs['patch_size']
        self.in_c = kwargs['in_chans']
        embed_dim = kwargs['embed_dim']
        depth = kwargs['depth']
        dropout = kwargs['drop_rate']

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

        # Added by Samar, need default pos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Add a projection for each skip connection
        self.use_add_skip = use_add_skip
        self.skip_idxs = {}
        skip_projs = {}
        for i in range(depth//2):
            # Only add skip connections every skip_rate blocks
            if (i + 1) % skip_rate == 0:
                # If num blocks between blocks is too small, then don't add skip
                if depth - 2*(i + 1) > skip_rate:
                    self.skip_idxs[i] = depth-(i+1)  # i is out connection, depth-(i+1) is in connection
                    skip_projs[depth-(i+1)] = vit.Mlp(in_features=2*embed_dim, out_features=embed_dim, drop=dropout) \
                        if not use_add_skip else nn.Identity()
        self.skip_projs = nn.ModuleDict({str(i): module for i, module in skip_projs.items()})
        print(f"Using skip connections: {self.skip_idxs}")

        self.decoder_pred = nn.Linear(embed_dim, self.patch_size ** 2 * self.in_c, bias=True)  # decoder to patch

        self.final_conv = nn.Conv2d(self.in_c, self.in_c, kernel_size=3, stride=1, padding='same') \
            if use_final_conv else nn.Identity()

        self.initialize_weights()

        # self.global_pool = global_pool
        # if self.global_pool:
        #     norm_layer = kwargs['norm_layer']
        #     embed_dim = kwargs['embed_dim']
        #     self.fc_norm = norm_layer(embed_dim)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed.requires_grad = False

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

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

    def forward(self, x, time):
        B = x.shape[0]
        x = self.patch_embed(x)  # (N, L, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # (N, L+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        """
        b1 -> x -> b2 -> x -> b3 -> x -> b4 -> x -> b5
              |          |          |          |
              |          ------------          |
              ----------------------------------
        """

        # embed time
        time = get_timestep_embedding(time, x.shape[-1])  # (N, D)
        time = time.unsqueeze(1)  # (N, 1, D)
        temb = self.temb(time)  # (N, 1, D_t)

        xs = {}
        for i, blk in enumerate(self.blocks):
            if i in xs:
                # This is for deeper layers to use previous xs
                prev_x = xs[i]
                if self.use_add_skip:
                    x = x + prev_x  # (N, L+1, D)
                else:
                    x = torch.cat((x, prev_x), dim=-1)  # (N, L+1, 2D)

                x = self.skip_projs[str(i)](x)  # (N, L+1, D)

            x = blk(x + self.temb_blocks[i](temb))  # (N, L+1, D)

            if i in self.skip_idxs:
                # Earlier layers save their xs
                end = self.skip_idxs[i]
                xs[end] = x

        x = self.norm(x)

        x = self.decoder_pred(x)  # (N, L+1, p^2 * C)

        # Remove cls token
        x = x[:, 1:, :]  # (N, L, p^2 * C)

        # Unpatchify
        x = self.unpatchify(x, self.patch_size, self.in_c)  # (N, C, H, W)

        # Final conv
        x = self.final_conv(x)  # (N, C, H, W)

        return x
