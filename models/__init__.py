from models.models_unet import UNet
from models.models_vit import VisionTransformer, ViTFinetune
from models.models_uvit import UVisionTransformer
from models.models_mae import MaskedAutoencoderViT
from models.models_umae import UMaskedAutoencoderViT


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
    elif config.model.type == "vit_mae" or config.model.type == "vit_mae_temb":
        img_size = config.data.image_size
        patch_size = config.model.patch_size
        in_channels = config.model.in_channels

        embed_dim = config.model.encoder.embed_dim
        depth = config.model.encoder.depth
        num_attn_heads = config.model.encoder.num_heads

        decoder_embed_dim = config.model.decoder.embed_dim
        decoder_depth = config.model.decoder.depth
        decoder_num_heads = config.model.decoder.num_heads

        temb_dim = config.model.temb_dim
        mlp_ratio = config.model.mlp_ratio
        dropout = config.model.dropout

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
            temb_dim=temb_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    elif config.model.type == 'umae':
        img_size = config.data.image_size
        patch_size = config.model.patch_size
        in_channels = config.model.in_channels

        embed_dim = config.model.encoder.embed_dim
        depth = config.model.encoder.depth
        num_attn_heads = config.model.encoder.num_heads

        decoder_embed_dim = config.model.decoder.embed_dim
        decoder_depth = config.model.decoder.depth
        decoder_num_heads = config.model.decoder.num_heads

        use_final_conv = config.model.use_final_conv
        skip_idxs = dict(config.model.skip_idxs)
        use_add_skip = config.model.use_add_skip

        temb_dim = config.model.temb_dim
        mlp_ratio = config.model.mlp_ratio
        dropout = config.model.dropout

        return UMaskedAutoencoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_attn_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            temb_dim=temb_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            skip_idxs=skip_idxs,
            use_add_skip=use_add_skip,
            use_final_conv=use_final_conv,
        )
    elif config.model.type == 'vit':
        img_size = config.data.image_size
        patch_size = config.model.patch_size
        in_channels = config.model.in_channels
        embed_dim = config.model.embed_dim
        depth = config.model.depth
        num_attn_heads = config.model.num_heads

        use_final_conv = config.model.use_final_conv
        is_finetune = config.model.finetune
        nb_classes = getattr(config.model, 'nb_classes', 0)

        temb_dim = config.model.temb_dim
        mlp_ratio = config.model.mlp_ratio
        dropout = config.model.dropout

        if is_finetune:
            use_temb = config.model.use_temb
            return ViTFinetune(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_channels,
                num_classes=nb_classes,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_attn_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=dropout,
                temb_dim=temb_dim,
                use_temb=use_temb,
            )

        return VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=0,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_attn_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=dropout,
            temb_dim=temb_dim,
            use_final_conv=use_final_conv,
        )
    elif config.model.type == 'uvit':
        img_size = config.data.image_size
        patch_size = config.model.patch_size
        in_channels = config.model.in_channels
        embed_dim = config.model.embed_dim
        depth = config.model.depth
        num_attn_heads = config.model.num_heads

        use_final_conv = config.model.use_final_conv
        skip_rate = config.model.skip_rate
        use_add_skip = config.model.use_add_skip

        temb_dim = config.model.temb_dim
        mlp_ratio = config.model.mlp_ratio
        dropout = config.model.dropout

        return UVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_attn_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=dropout,
            temb_dim=temb_dim,
            use_add_skip=use_add_skip,
            skip_rate=skip_rate,
            use_final_conv=use_final_conv,
        )
    else:
        raise NotImplementedError("Wrong model type")
