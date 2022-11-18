import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def snr_image_loss(
        model,
        x0: torch.Tensor,
        t: torch.LongTensor,
        e: torch.Tensor,
        b: torch.Tensor, keepdim=False
):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())

    if keepdim:
        return (a/(1.0-a)).clamp(min=1.0) * (x0 - output).pow(2).sum(dim=(1, 2, 3))
    else:
        return (a/(1.0-a)).clamp(min=1.0) * (x0 - output).pow(2).sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
    'unet': noise_estimation_loss,
    'vit': noise_estimation_loss,
    'vit_mae': noise_estimation_loss,
    'vit_mae_temb': noise_estimation_loss,
    'umae': snr_image_loss,
    'uvit': noise_estimation_loss,
}
