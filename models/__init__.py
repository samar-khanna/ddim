from models.models_unet import UNet
from models.models_mae import MaskedAutoencoderViT
from models.models_vit import VisionTransformer
from models.segm.model.factory import create_segmenter

from models.models_umae import UMaskedAutoencoderViT

from ddim.models.models_vit import ViTFinetune


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
    elif config.model.type in ['umae', 'umae_x']:
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
        in_chans = config.model.in_channels
        num_classes = config.model.num_classes,
        # global_pool = 'token',
        embed_dim = config.model.embed_dim
        depth = config.model.depth
        num_heads = config.model.num_heads
        mlp_ratio = config.model.mlp_ratio
        is_finetune = config.model.finetune
        temb_dim = config.model.temb_dim
        dropout = config.model.dropout


        # qkv_bias = True,
        # init_values = None,
        # class_token = True,
        # no_embed_class = False,
        # pre_norm = False,
        # fc_norm = None,
        # drop_rate = 0.,
        # attn_drop_rate = 0.,
        # drop_path_rate = 0.,
        # weight_init = '',
        # embed_layer = PatchEmbed,
        # norm_layer = None,
        # act_layer = None,
        # block_fn = Block,
        if is_finetune:
            use_temb = config.model.use_temb
            return ViTFinetune(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=num_classes,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,

                mlp_ratio=mlp_ratio,
                drop_rate=dropout,
                temb_dim=temb_dim,
                use_temb=use_temb,
            )
        return VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
        )
    elif config.model.type == "vit_segm":
        # img_size = config.data.image_size
        # patch_size = config.model.patch_size
        # in_channels = config.model.in_channels
        #
        # embed_dim = config.model.encoder.embed_dim
        # depth = config.model.encoder.depth
        # num_attn_heads = config.model.encoder.num_heads
        #
        # decoder_embed_dim = config.model.decoder.embed_dim
        # decoder_depth = config.model.decoder.depth
        # decoder_num_heads = config.model.decoder.num_heads

        # mlp_ratio = config.model.mlp_ratio
        model_cfg = dict()
        decoder = dict()
        decoder['name'] = config.model.decoder.name
        decoder['dropout'] =config.model.decoder.mask_transformer.dropout
        decoder['drop_path_rate'] =config.model.decoder.mask_transformer.drop_path_rate
        decoder['n_layers'] =config.model.decoder.mask_transformer.n_layers
        # decoder['d_encoder']=
        # decoder['patch_size']=
        model_cfg["n_cls"]=config.model.n_cls
        model_cfg["image_size"] = (config.data.image_size, config.data.image_size)
        model_cfg["patch_size"]=config.model.patch_size
        model_cfg["d_model"]=config.model.d_model
        model_cfg["n_heads"]=config.model.n_heads
        model_cfg["n_layers"]=config.model.n_layers
        
        model_cfg["backbone"] = config.model.backbone
        model_cfg["dropout"] = config.model.dropout
        model_cfg["drop_path_rate"] = config.model.drop_path_rate
        model_cfg["decoder"] = decoder
        model_cfg["normalization"]=config.model.normalization

        # model_cfg["normalization"]

        model = create_segmenter(model_cfg)
        return model
    else:
        raise NotImplementedError("Wrong model type")
