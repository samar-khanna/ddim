from models.models_unet import UNet
from models.models_mae import MaskedAutoencoderViT


def get_model(config):
    if config.model.type == "unet" or config.model.type == "simple":
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps

        return UNet(
            img_size=resolution,
            in_ch=in_channels,
            dim=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resample_with_conv=resamp_with_conv
        )
    elif config.model.type == "vit_mae":
        img_size = config.data.image_size
        patch_size = config.model.patch_size
        in_channels = config.model.in_channels

        embed_dim = config.model.encoder.embed_dim
        depth = config.model.encoder.depth
        num_attn_heads = config.model.encoder.num_heads

        decoder_embed_dim = config.model.decoder.embed_dim
        decoder_depth = config.model.decoder.depth
        decoder_num_heads = config.model.decoder.num_heads

        mlp_ratio = config.model.mlp_ratio

        return MaskedAutoencoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_attn_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
        )
    else:
        raise NotImplementedError("Wrong model type")
