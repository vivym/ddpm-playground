import math
import random
from collections import namedtuple
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import reduce

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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
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

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

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

    def training_step(self, batch, batch_idx: int):
        loss = self.forward(batch)

        self.log("train/loss", loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        print("validation_step")

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
