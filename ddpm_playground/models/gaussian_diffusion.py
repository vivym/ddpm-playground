import math
import random
from collections import namedtuple
from typing import Optional, Tuple
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import reduce
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
from PIL import Image

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def linear_beta_schedule(timesteps: int):
    scale = 1000. / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(
        beta_start, beta_end, timesteps, dtype=torch.float64
    )


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0., 0.999)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        image_size: int = 128,
        timesteps: int = 1000,
        sampling_timesteps: Optional[int] = None,
        loss_type: str = "l1",  # l1 or l2
        objective: str = "pred_noise",  # pred_noise, pred_x0, pred_v
        beta_schedule: str = "cosine",  # cosine, linear
        # p2 loss weight, from https://arxiv.org/abs/2204.00227
        # 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_gamma: float = 0.,
        p2_loss_weight_k: float = 1.,
        ddim_sampling_eta: float = 1.,
        learning_rate: float = 1e-3,
        num_val_samples: int = 32,
        num_test_samples: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.in_channels = model.in_channels
        self.self_condition = model.self_condition

        self.image_size = image_size

        assert loss_type in ["l1", "l2"]
        self.loss_type = loss_type

        assert objective in ["pred_noise", "pred_x0", "pred_v"], (
            "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v "
            "(predict v [v-parameterization as defined in appendix D of progressive distillation paper, "
            "used in imagen-video successfully])"
        )
        self.objective = objective

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps = betas.shape[0]
        self.timesteps = timesteps

        if sampling_timesteps is None:
            sampling_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer_f32 = lambda name, buf: self.register_buffer(name, buf.to(torch.float32))

        register_buffer_f32("betas", betas)
        register_buffer_f32("alphas_cumprod", alphas_cumprod)
        register_buffer_f32("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer_f32("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer_f32("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        register_buffer_f32("log_one_minus_alphas_cumprod", torch.log(1. - alphas_cumprod))
        register_buffer_f32("sqrt_recip_alphas_cumprod", torch.sqrt(1. / alphas_cumprod))
        register_buffer_f32("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer_f32("posterior_variance", posterior_variance)

        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer_f32(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer_f32(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
        )
        register_buffer_f32(
            "posterior_mean_coef2",
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod),
        )

        # calculate p2 reweighting

        register_buffer_f32(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma,
        )

        self.learning_rate = learning_rate
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, x_self_cond)
        identity = lambda x: x
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = x_start.shape

        if noise is None:
            noise = torch.randn_like(x_start)

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        else:
            x_self_cond = None

        model_out = self.model(x, t, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, x: torch.Tensor):
        batch_size, _, height, width = x.shape
        assert height == self.image_size and width == self.image_size
        device = x.device

        t = torch.randint(
            0, self.timesteps, (batch_size,), dtype=torch.int64, device=device
        )

        x = x * 2 - 1
        return self.p_losses(x, t)

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: int,
        x_self_cond: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        x_self_cond: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ):
        batched_times = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_imgs = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_imgs, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int, int, int, int]):
        batch, device = shape[0], self.betas.device

        imgs = torch.randn(shape, device=device)

        x_start = None

        for t in reversed(range(0, self.timesteps)):
            self_cond = x_start if self.self_condition else None
            imgs, x_start = self.p_sample(imgs, t, x_self_cond=self_cond)

        imgs = (imgs + 1) * 0.5
        return imgs

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple[int, int, int, int],
        clip_denoised: bool = True
    ):
        batch, device = shape[0], self.betas.device
        total_timesteps, sampling_timesteps = self.timesteps, self.sampling_timesteps
        eta, objective = self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        imgs = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                imgs, time_cond, self_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                imgs = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(imgs)

            imgs = x_start * alpha_next.sqrt() + \
                   c * pred_noise + \
                   sigma * noise

        imgs = (imgs +  1) * 0.5
        return imgs

    @torch.no_grad()
    def sample(self, batch_size: int = 16):
        shape = (batch_size, self.in_channels, self.image_size, self.image_size)
        if self.is_ddim_sampling:
            return self.ddim_sample(shape)
        else:
            return self.p_sample_loop(shape)

    def training_step(self, batch, batch_idx: int):
        loss = self.forward(batch)

        self.log("train/loss", loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        imgs = self.sample(self.num_val_samples)

        if self.logger and isinstance(self.logger, WandbLogger):
            images = []
            for img in imgs:
                grid = make_grid(img)
                grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
                ndarr = grid.to("cpu", torch.uint8).numpy()
                images.append(Image.fromarray(ndarr))
            self.logger.log_image(key="val/sample", images=images)

    def test_step(self, batch, batch_idx: int):
        num_batches = math.ceil(self.num_test_samples / 16)

        save_dir = Path("results")
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        for i in range(num_batches):
            imgs = self.sample(16)

            for j, img in enumerate(imgs):
                grid = make_grid(img)
                grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
                ndarr = grid.to("cpu", torch.uint8).numpy()
                img = Image.fromarray(ndarr)
                img.save(save_dir / f"{i * 16 + j:08d}.png")

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
