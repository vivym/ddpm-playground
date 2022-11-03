import math
from typing import Optional, List, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device

        half_channels = self.num_channels // 2
        emb = math.log(10000) / (half_channels - 1)
        emb = torch.exp(torch.arange(half_channels, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, num_channels: int, is_random: bool = False):
        super().__init__()
        assert (num_channels % 2) == 0
        half_channels = num_channels // 2
        self.weights = nn.Parameter(
            torch.randn(half_channels), requires_grad = not is_random
        )

    def forward(self, x):
        x = x[:, None]
        freqs = x * self.weights[None, :] * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x, normalized_weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups,
        )


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
    ):
        super().__init__()

        self.proj = WeightStandardizedConv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: Optional[int] = None,
        num_groups: int = 8,
    ):
        super().__init__()

        if time_channels is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_channels, out_channels * 2)
            )

        self.block1 = ConvBlock(in_channels, out_channels, num_groups=num_groups)
        self.block2 = ConvBlock(out_channels, out_channels, num_groups=num_groups)

        if in_channels != out_channels:
            self.short_cut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.short_cut = nn.Identity()

    def forward(self, x, time_embd = None):
        if self.time_mlp is not None and time_embd is not None:
            time_embd = self.time_mlp(time_embd)
            time_embd = time_embd[:, :, None, None]
            scale_shift = time_embd.chunk(2, dim=1)
        else:
            scale_shift = None

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.short_cut(x)


class LayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, num_channels, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class LinearAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        hidden_channels: int = 128,
    ):
        super().__init__()

        self.num_heads = num_heads
        assert hidden_channels % num_heads == 0
        self.scale = (hidden_channels / num_heads) ** -0.5

        self.qkv_proj = nn.Conv2d(
            in_channels, hidden_channels * 3, kernel_size=1, bias=False,
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            LayerNorm(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        qkv = self.qkv_proj(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.num_heads),
            qkv,
        )

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.num_heads, x=h, y=w)

        return self.out_proj(out)


class Attention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        hidden_channels: int = 128,
    ):
        super().__init__()

        self.num_heads = num_heads
        assert hidden_channels % num_heads == 0
        self.scale = (hidden_channels / num_heads) ** -0.5

        self.qkv_proj = nn.Conv2d(
            in_channels, hidden_channels * 3, kernel_size=1, bias=False,
        )
        self.out_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.num_heads),
            qkv,
        )

        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        return self.out_proj(out)


class PreNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        fn: nn.Module,
    ):
        super().__init__()

        self.fn = fn
        self.norm = LayerNorm(in_channels)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()

        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(in_channels: int, out_channels: Optional[int] = None):
    out_channels = out_channels if out_channels is not None else in_channels

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode = "nearest"),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )

def Downsample(in_channels: int, out_channels: Optional[int] = None):
    out_channels = out_channels if out_channels is not None else in_channels

    return nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_channels: int = 64,
        channel_multiplier: Tuple[int, ...] = (1, 2, 4, 8),
        self_condition: bool = False,
        time_embedding: str = "sinusoidal", # sinusoidal, learned_sinusoidal, random_sinusoidal
        learned_sinusoidal_channels: int = 16,
        num_res_block_groups: int = 8,
        learned_variance: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        if learned_variance:
            self.out_channels *= 2

        self.self_condition = self_condition

        if self.self_condition:
            in_channels *= 2

        self.stem = nn.Conv2d(
            in_channels, num_channels,
            kernel_size=7, padding=3,
        )

        time_channels = num_channels * 4

        if time_embedding == "sinusoidal":
            pos_embd = SinusoidalPosEmb(num_channels)
            fourier_channels = num_channels
        elif time_embedding == "learned_sinusoidal":
            pos_embd = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_channels, False)
            fourier_channels = learned_sinusoidal_channels + 1
        elif time_embedding == "random_sinusoidal":
            pos_embd = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_channels, True)
            fourier_channels = learned_sinusoidal_channels + 1
        else:
            raise ValueError(f"Unknown time_embedding: {time_embedding}")

        self.time_mlp = nn.Sequential(
            pos_embd,
            nn.Linear(fourier_channels, time_channels),
            nn.GELU(),
            nn.Linear(time_channels, time_channels)
        )

        channels_list = [num_channels, *map(lambda m: num_channels * m, channel_multiplier)]
        in_out_channels_list = list(zip(channels_list[:-1], channels_list[1:]))
        num_stages = len(in_out_channels_list)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        residual_block = partial(
            ResidualBlock, time_channels=time_channels, num_groups=num_res_block_groups,
        )

        for stage, (in_channels, out_channels) in enumerate(in_out_channels_list):
            is_last_stage = stage == (num_stages - 1)

            stage_layers = nn.ModuleList([
                residual_block(in_channels, in_channels),
                residual_block(in_channels, in_channels),
                Residual(PreNorm(in_channels, LinearAttention(in_channels))),
            ])

            if is_last_stage:
                stage_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                stage_layers.append(Downsample(in_channels, out_channels))

            self.downs.append(stage_layers)

        mid_channels = channels_list[-1]
        self.mid_block1 = residual_block(mid_channels, mid_channels)
        self.mid_attn = Residual(PreNorm(mid_channels, Attention(mid_channels)))
        self.mid_block2 = residual_block(mid_channels, mid_channels)

        for stage, (in_channels, out_channels) in enumerate(reversed(in_out_channels_list)):
            is_last_stage = stage == (num_stages - 1)

            stage_layers = nn.ModuleList([
                residual_block(in_channels + out_channels, out_channels),
                residual_block(in_channels + out_channels, out_channels),
                Residual(PreNorm(out_channels, LinearAttention(out_channels))),
            ])

            if is_last_stage:
                stage_layers.append(nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1))
            else:
                stage_layers.append(Upsample(out_channels, in_channels))

            self.ups.append(stage_layers)

        self.out_block = residual_block(num_channels * 2, num_channels)
        self.out_conv = nn.Conv2d(num_channels, self.out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        x_self_cond: Optional[torch.Tensor] = None
    ):
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
                x = torch.cat((x_self_cond, x), dim = 1)

        x = self.stem(x)
        r = x

        t = self.time_mlp(time)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.out_block(x, t)
        return self.out_conv(x)
