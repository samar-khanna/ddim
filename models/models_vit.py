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


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,  temb_dim=0, use_final_conv=True, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.patch_size = kwargs['patch_size']
        self.in_c = kwargs['in_chans']
        embed_dim = kwargs['embed_dim']
        depth = kwargs['depth']
        dropout = kwargs['drop_rate']

        # Added by Samar, need default pos embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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

        self.decoder_pred = nn.Linear(embed_dim, self.patch_size ** 2 * self.in_c, bias=True)  # decoder to patch

        self.final_conv = nn.Conv2d(self.in_c, self.in_c, kernel_size=3, stride=1, padding='same') \
            if use_final_conv else nn.Identity()

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

        # embed time
        time = get_timestep_embedding(time, x.shape[-1])  # (N, D)
        time = time.unsqueeze(1)  # (N, 1, D)
        temb = self.temb(time)  # (N, 1, D_t)

        for i, blk in enumerate(self.blocks):
            x = blk(x + self.temb_blocks[i](temb))  # (N, L+1, D)

        x = self.norm(x)

        x = self.decoder_pred(x)  # (N, L+1, p^2 * C)

        # Remove cls token
        x = x[:, 1:, :]  # (N, L, p^2 * C)

        # Unpatchify
        x = self.unpatchify(x, self.patch_size, self.in_c)  # (N, C, H, W)

        # Final conv
        x = self.final_conv(x)  # (N, C, H, W)

        return x


class ViTFinetune(VisionTransformer):
    def __init__(self, use_temb=False, num_timesteps=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_timesteps = num_timesteps
        self.use_temb = use_temb
        if not self.use_temb:
            del self.temb
            del self.temb_blocks

        norm_layer = nn.LayerNorm
        embed_dim = kwargs['embed_dim']
        self.fc_norm = norm_layer(embed_dim)

        del self.decoder_pred
        del self.final_conv

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # (N, L+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # embed time
        if self.use_temb:
            t = torch.zeros(B//2 + 1, device=x.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:B]
            time = get_timestep_embedding(t, x.shape[-1])  # (N, D)
            time = time.unsqueeze(1)  # (N, 1, D)
            temb = self.temb(time)  # (N, 1, D_t)

        for i, blk in enumerate(self.blocks):
            if self.use_temb:
                x = blk(x + self.temb_blocks[i](temb))  # (N, L+1, D)
            else:
                x = blk(x)

        x = self.norm(x)

        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        x = self.fc_norm(x)  # (N, D)

        outcome = self.head(x)  # (N, #classes)
        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model