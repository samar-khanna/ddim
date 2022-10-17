import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segm.model.utils import padding, unpadding
from timm.models.layers import trunc_normal_
import math


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Segmenter(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.in_chans=3
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.img_restore = nn.Linear(self.encoder.d_model, self.patch_size ** 2 * self.in_chans, bias=True)  # decoder to patch



    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params
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
    def forward(self, im, time):
       # time = time.unsqueeze(1)  # (N, 1, D)
        H_ori, W_ori = im.size(2), im.size(3)

        #im = padding(im, self.patch_size)
        print('shape without padding:',im.shape)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)
        time = get_timestep_embedding(time, x.shape[-1])  # (N, D)
        time = time.unsqueeze(1)  # (N, 1, D)

        print('after encoder in segmenter shape: ', x.shape)
        x = x + time
        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))
        x = self.img_restore(masks)  # N, L, P*P*C
        # unpatchify

        x = self.unpatchify(x, self.patch_size, self.in_chans)  # (N, C, H, W)
        print('in segmenter after decoder:', masks.shape)
        #masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        print('in segmenter after unpatchify:', x.shape)

        #masks = unpadding(masks, (H_ori, W_ori))
        #print('in segmenter after unpadding:', masks.shape)


        return x

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
