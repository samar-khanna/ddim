# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

from models.pos_embed import get_2d_sincos_pos_embed
from models.time_embed import get_timestep_embedding, ZerosLike


class UVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, temb_dim=0, **kwargs):
        super().__init__(**kwargs)

        self.patch_size = kwargs['patch_size']
        self.in_c = kwargs['in_chans']
        embed_dim = kwargs['embed_dim']
        depth = kwargs['depth']

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
        self.skip_projs = nn.ModuleList([
            nn.Linear(2*self.embed_dim, self.embed_dim, bias=True)
            for _ in range(len(self.blocks)//2)
        ])

        self.decoder_pred = nn.Linear(embed_dim, self.patch_size ** 2 * self.in_c, bias=True)  # decoder to patch

        self.final_conv = nn.Conv2d(self.in_c, self.in_c, kernel_size=3, stride=1, padding='same')

        # self.global_pool = global_pool
        # if self.global_pool:
        #     norm_layer = kwargs['norm_layer']
        #     embed_dim = kwargs['embed_dim']
        #     self.fc_norm = norm_layer(embed_dim)

        del self.norm  # remove the original norm

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
        x = self.patch_embed(x)

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

        t_idx = 0
        num_low_res_blocks = len(self.blocks)//2
        xs = []
        for blk in self.blocks[:num_low_res_blocks]:
            x = blk(x + self.temb_blocks[t_idx](temb))  # (N, L+1, D)
            xs.append(x)
            t_idx += 1

        # Reverse xs as we want lowest res last
        xs = xs[::-1]

        # mid block
        x = self.blocks[num_low_res_blocks](x + self.temb_blocks[t_idx](temb))  # (N, L+1, D)
        t_idx += 1

        # up blocks
        for i, blk in enumerate(self.blocks[num_low_res_blocks+1:]):
            prev_x = xs[i]
            x = torch.cat((x, prev_x), dim=-1)  # (N, L+1, 2D)
            x = self.skip_projs[i](x)  # (N, L+1, D)

            x = blk(x + self.temb_blocks[t_idx](temb))  # (N, L+1, D)
            t_idx += 1

        x = self.decoder_pred(x)  # (N, L+1, p^2 * C)

        # Remove cls token
        x = x[:, 1:, :]  # (N, L, p^2 * C)

        # Unpatchify
        x = self.unpatchify(x, self.patch_size, self.in_c)  # (N, C, H, W)

        # Final conv
        x = self.final_conv(x)  # (N, C, H, W)

        return x
