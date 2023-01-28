import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def imagen_pixel_clip(x0_pred):
    dims = x0_pred.dim()
    p = 0.995  # A hyperparameter in the paper of "Imagen" [1].
    scale = torch.quantile(torch.abs(x0_pred).reshape((x0_pred.shape[0], -1)), p, dim=1)
    scale = expand_dims(torch.maximum(scale, 1. * torch.ones_like(scale).to(scale.device)), dims)
    x0_pred = torch.clamp(x0_pred, -scale, scale) / scale
    return x0_pred


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.
    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]


def generalized_steps(x, seq, model, noise_schedule, final_denoise, **kwargs):
    ns = noise_schedule
    with torch.no_grad():
        n = x.size(0)
        dims = x.dim()

        x0_preds = []
        xs = [x]
        for s, t in zip(seq, seq[1:]):
            s_vec = (torch.ones(n) * s.item()).to(x.device)
            t_vec = (torch.ones(n) * t.item()).to(x.device)

            lambda_s, lambda_t = ns.marginal_lambda(s_vec), ns.marginal_lambda(t_vec)
            h = lambda_t - lambda_s
            log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s_vec), ns.marginal_log_mean_coeff(t_vec)
            sigma_s, sigma_t = ns.marginal_std(s_vec), ns.marginal_std(t_vec)
            alpha_t = torch.exp(log_alpha_t)

            phi_1 = torch.expm1(h)

            x_prev = xs[-1].to(x.device)
            # model_s = model(x_prev, (s_vec + 1.)/ns.total_N)  # Note: scale s to between 0 and total_N for backward compatibility
            model_s = model(x_prev, s_vec)
            x_next = (
                    expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims) * x_prev
                    - expand_dims(sigma_t * phi_1, dims) * model_s
            )

            xs.append(x_next.to('cpu'))

        if final_denoise:
            t_vec = (torch.ones(n) * seq[-1].item()).to(x.device)  # t0

            x_prev = xs[-1].to(x.device)
            # eps_pred = model(x_prev, (t_vec + 1.) / ns.total_N)  # Note: scale s to between 0 and total_N for backward compatibility
            eps_pred = model(x_prev, t_vec)
            alpha_t, sigma_t = ns.marginal_alpha(t), ns.marginal_std(t)
            x0_pred = (x_prev - expand_dims(sigma_t, dims) * eps_pred) / expand_dims(alpha_t, dims)

            xs.append(x0_pred.to('cpu'))

        # with torch.no_grad():
        #     n = x.size(0)
        #     dims = x.dim()
        #     seq_next = [-1] + list(seq[:-1])
        #
        #     x0_preds = []
        #     xs = [x]
        #     for i, j in zip(reversed(seq), reversed(seq_next)):
        #         t = (torch.ones(n) * i).to(x.device)
        #         next_t = (torch.ones(n) * j).to(x.device)
        #         at = ns.marginal_alpha((t + 1.) / ns.total_N).pow(2).view(-1, 1, 1, 1)
        #         at_next = ns.marginal_alpha((next_t + 1.) / ns.total_N).pow(2).view(-1, 1, 1, 1)
        #         # at = compute_alpha(b, t.long())
        #         # at_next = compute_alpha(b, next_t.long())
        #         xt = xs[-1].to('cuda')
        #         et = model(xt, t)
        #         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        #         x0_preds.append(x0_t.to('cpu'))
        #         c1 = (
        #                 kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        #         )
        #         c2 = ((1 - at_next) - c1 ** 2).sqrt()
        #         xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        #         xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def generalized_image_steps(x, seq, model, noise_schedule, final_denoise, **kwargs):
    ns = noise_schedule
    with torch.no_grad():
        n = x.size(0)
        dims = x.dim()

        x0_preds = []
        xs = [x]
        for s, t in zip(seq, seq[1:]):
            s_vec = (torch.ones(n) * s.item()).to(x.device)
            t_vec = (torch.ones(n) * t.item()).to(x.device)

            lambda_s, lambda_t = ns.marginal_lambda(s_vec), ns.marginal_lambda(t_vec)
            h = lambda_t - lambda_s
            log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s_vec), ns.marginal_log_mean_coeff(t_vec)
            sigma_s, sigma_t = ns.marginal_std(s_vec), ns.marginal_std(t_vec)
            alpha_t = torch.exp(log_alpha_t)

            phi_1 = torch.expm1(-h)

            x_prev = xs[-1].to(x.device)
            model_s = model(x_prev, s_vec)
            model_s = imagen_pixel_clip(model_s)

            x_next = (
                    expand_dims(sigma_t / sigma_s, dims) * x_prev
                    - expand_dims(alpha_t * phi_1, dims) * model_s
            )
            xs.append(x_next.to('cpu'))

        if final_denoise:
            t_vec = (torch.ones(n) * seq[-1].item()).to(x.device)  # t0

            x_prev = xs[-1].to(x.device)
            x0_pred = model(x_prev, t_vec)
            x0_pred = imagen_pixel_clip(x0_pred)
            xs.append(x0_pred.to('cpu'))

        # seq_next = [-1] + list(seq[:-1])
        # for i, j in zip(reversed(seq), reversed(seq_next)):
        #     t = (torch.ones(n) * i).to(x.device)
        #     next_t = (torch.ones(n) * j).to(x.device)
        #     at = ns.marginal_alpha((t + 1.)/ns.total_N).pow(2)
        #     at_next = ns.marginal_alpha((next_t + 1.)/ns.total_N).pow(2)
        #     xt = xs[-1].to('cuda')
        #     x0_t = model(xt, (t + 1.)/ns.total_N)  # !!!! Note: divide t by total steps
        #     et = (xt - at.sqrt() * x0_t)/(1 - at).sqrt()
        #     x0_preds.append(x0_t.to('cpu'))
        #     c1 = (
        #         kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        #     )
        #     c2 = ((1 - at_next) - c1 ** 2).sqrt()
        #     xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        #     xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, noise_schedule, **kwargs):
    ns = noise_schedule
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = ns.marginal_alpha((t + 1.)/ns.total_N).pow(2)
            atm1 = ns.marginal_alpha((next_t + 1.)/ns.total_N).pow(2)
            # at = compute_alpha(betas, t.long())
            # atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
