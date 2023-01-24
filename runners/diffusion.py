import os
import logging
import time
import glob

import numpy as np
import wandb
import tqdm
import torch
import torch.utils.data as data

from models import get_model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from runners.noise_schedule import NoiseScheduleVP, betas_for_alpha_bar

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.as_tensor(betas))
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.as_tensor(betas))
    elif beta_schedule == 'cosine':
        betas = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
        noise_schedule = NoiseScheduleVP(schedule='cosine')
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.as_tensor(betas))
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.as_tensor(betas))
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.as_tensor(betas))
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas, noise_schedule


def get_time_steps(noise_schedule, skip_type, t_T, t_0, N, device):
    """Compute the intermediate time steps for sampling.
    Args:
        skip_type: A `str`. The type for the spacing of the time steps. We support three types:
            - 'logSNR': uniform logSNR for the time steps.
            - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
            - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
        t_T: A `float`. The starting time of the sampling (default is T).
        t_0: A `float`. The ending time of the sampling (default is epsilon).
        N: A `int`. The total number of the spacing of the time steps.
        device: A torch device.
    Returns:
        A pytorch tensor of the time steps, with the shape (N + 1,).
    """
    if skip_type == 'logSNR':
        lambda_T = noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
        lambda_0 = noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
        logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
        return noise_schedule.inverse_lambda(logSNR_steps)
    elif skip_type == 'time_uniform' or skip_type == 'uniform':
        return torch.linspace(t_T, t_0, N + 1).to(device)
    elif skip_type == 'time_quadratic':
        t_order = 2
        t = torch.linspace(t_T ** (1. / t_order), t_0 ** (1. / t_order), N + 1).pow(t_order).to(device)
        return t
    else:
        raise ValueError(
            "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas, noise_schedule = get_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.noise_schedule = noise_schedule

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = get_model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        # Get model info
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info('number of params (M): %.2f' % (n_parameters / 1.e6))

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        # Watch model (W&B)
        if self.args.wandb is not None:
            wandb.watch(model)

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.rand(size=(n // 2 + 1,)).to(self.device)
                # t = torch.randint(
                #     low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                # ).to(self.device)
                t = torch.cat([t, (1.0 - t).clamp(max=0.999)], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, self.noise_schedule)

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                # Log info
                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                if self.args.wandb is not None:
                    wandb.log({'Train/loss': loss.item(), 'Train/step': step})

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        model = get_model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for i in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                tvu.save_image(x, os.path.join(self.args.image_folder, f"image_grid{i}.png"), n_row=int(np.sqrt(x.shape[0])))
                # for i in range(n):
                #     tvu.save_image(
                #         x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                #     )
                #     img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        t_T = self.noise_schedule.T
        t_0 = 1. / self.noise_schedule.total_N if self.noise_schedule.schedule == 'discrete' else 1e-4
        t_seq = get_time_steps(self.noise_schedule, self.args.skip_type, t_T, t_0, self.args.timesteps, self.device)

        if self.args.sample_type == "generalized":

            # if self.args.skip_type == "uniform":
            #     skip = self.num_timesteps // self.args.timesteps
            #     t_seq = range(0, self.num_timesteps, skip)
            # elif self.args.skip_type == "quad":
            #     t_seq = (
            #         np.linspace(
            #             0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
            #         )
            #         ** 2
            #     )
            #     t_seq = [int(s) for s in list(t_seq)]
            # else:
            #     raise NotImplementedError

            from functions.denoising import generalized_steps

            xs = generalized_steps(
                x, t_seq, model, self.noise_schedule, self.args.final_denoise, eta=self.args.eta
            )
            x = xs
        elif self.args.sample_type == "generalized_image":
            from functions.denoising import generalized_image_steps

            xs = generalized_image_steps(
                x, t_seq, model, self.noise_schedule, self.args.final_denoise, eta=self.args.eta
            )
            x = xs

        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                t_seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                t_seq = [int(s) for s in list(t_seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, t_seq, model, self.noise_schedule)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
