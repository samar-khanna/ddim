import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          noise_schedule, keepdim=False):
    # a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    a = noise_schedule.marginal_alpha(t).view(-1, 1, 1, 1)
    sigma = noise_schedule.marginal_std(t).view(-1, 1, 1, 1)
    x = x0 * a + e * sigma
    # output = model(x, t.float() * noise_schedule.total_N - 1)  # Note: scale T to between 0 and total_N for backward compatibility
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().mean(dim=(1, 2, 3))
    else:
        return (e - output).square().mean(dim=(1, 2, 3)).mean(dim=0)


def snr_image_loss(
        model,
        x0: torch.Tensor,
        t: torch.LongTensor,
        e: torch.Tensor,
        noise_schedule, keepdim=False
):
    # a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    # Ref: https://github.com/google-research/google-research/blob/32363faa531d8386a5733b54803893f957472e1e/diffusion_distillation/diffusion_distillation/dpm.py#L222
    logsnr = 2.0 * noise_schedule.marginal_lambda(t).view(-1, 1, 1, 1)
    a = logsnr.sigmoid().sqrt()
    sigma = (-logsnr).sigmoid().sqrt()
    x = x0 * a + e * sigma

    x0_pred = model(x, t.float())
    # eps = (x_t - alpha*x0)/sigma.
    eps_pred = (1. + logsnr.exp()).sqrt() * (x - x0 * (1. + (-logsnr).exp()).rsqrt())

    x_mse = (x0_pred - x0).pow(2).mean(dim=(1, 2, 3))  # (N,)
    eps_mse = (eps_pred - e).pow(2).mean(dim=(1, 2, 3))  # (N,)
    batch_loss = torch.maximum(x_mse, eps_mse)

    # a = a.view(-1)
    # sigma = sigma.view(-1)
    # batch_loss = (a.pow(2)/sigma.pow(2)).clamp(min=1.0) * (x0 - x0_pred).pow(2).sum(dim=(1, 2, 3))  # (N,)
    if keepdim:
        return batch_loss
    else:
        return batch_loss.mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
    'unet': noise_estimation_loss,
    'unet_x': snr_image_loss,
    'vit': noise_estimation_loss,
    'vit_mae': noise_estimation_loss,
    'vit_mae_temb': noise_estimation_loss,
    'umae': noise_estimation_loss,
    'umae_x': snr_image_loss,
    'uvit': noise_estimation_loss,
    'ddpmae': snr_image_loss,
}
