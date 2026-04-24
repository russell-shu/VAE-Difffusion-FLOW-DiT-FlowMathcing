"""Wav-VAE building blocks.

Implements the three ingredients of LongCat-AudioDiT section 3.1:

* Snake activation (Ziyin 2020, learnable per-channel ``alpha``).
* Dilated residual unit (paper eq. 1).
* Oobleck-style encoder / decoder blocks with a non-parametric
  space<->channel + channel-averaging shortcut.

The non-parametric shortcut is a *linear* residual pathway that bypasses
the Snake / conv nonlinearities so the autoencoder can stay stable
under aggressive downsampling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class Snake(nn.Module):
    """Snake activation: ``x + (1/alpha) * sin(alpha * x) ** 2``.

    A per-channel learnable ``alpha`` lets the network adapt the
    periodicity of the activation to the acoustic content.  We clamp
    ``alpha`` above ``eps`` to avoid dividing by zero.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0, eps: float = 1e-9) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1), alpha_init))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha + self.eps
        return x + (1.0 / alpha) * torch.sin(alpha * x).pow(2)


class DilatedResUnit(nn.Module):
    """Paper eq. 1: ``h <- h + Conv_1x1(sigma(Conv_{k,d}(sigma(h))))``."""

    def __init__(self, channels: int, kernel: int = 7, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.act1 = Snake(channels)
        self.conv1 = weight_norm(
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation)
        )
        self.act2 = Snake(channels)
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(x))
        h = self.conv2(self.act2(h))
        return x + h


def space_to_channel(x: torch.Tensor, stride: int) -> torch.Tensor:
    """``[B, C, T]`` -> ``[B, C * stride, T // stride]``.

    Equivalent to einops ``rearrange('b c (n s) -> b (c s) n', s=stride)``
    after trimming ``T`` down to a multiple of ``stride``.
    """
    B, C, T = x.shape
    if T % stride != 0:
        x = x[..., : T - (T % stride)]
        T = x.shape[-1]
    x = x.reshape(B, C, T // stride, stride)
    x = x.permute(0, 1, 3, 2).contiguous()
    return x.reshape(B, C * stride, T // stride)


def channel_to_space(x: torch.Tensor, stride: int) -> torch.Tensor:
    """Inverse of :func:`space_to_channel`.  Assumes ``C % stride == 0``."""
    B, C, T = x.shape
    assert C % stride == 0, f"channel_to_space needs C={C} divisible by stride={stride}"
    C_out = C // stride
    x = x.reshape(B, C_out, stride, T)
    x = x.permute(0, 1, 3, 2).contiguous()
    return x.reshape(B, C_out, T * stride)


def match_channels(x: torch.Tensor, target: int) -> torch.Tensor:
    """Reduce / expand channel count to ``target`` via grouped mean or
    replication-and-average.  Non-parametric.
    """
    B, C, T = x.shape
    if C == target:
        return x
    if C % target == 0:
        group = C // target
        return x.reshape(B, target, group, T).mean(dim=2)
    if target % C == 0:
        reps = target // C
        return x.repeat_interleave(reps, dim=1)
    # Fallback: adaptive averaging across the channel dim.
    x_t = x.transpose(1, 2)
    x_t = F.adaptive_avg_pool1d(x_t, target)
    return x_t.transpose(1, 2)


class EncoderBlock(nn.Module):
    """A stack of dilated residual units followed by a strided conv
    downsample, with a non-parametric shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations: tuple[int, ...] = (1, 3, 9),
        kernel: int = 7,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.units = nn.ModuleList(
            [DilatedResUnit(in_channels, kernel=kernel, dilation=d) for d in dilations]
        )
        self.act = Snake(in_channels)
        self.down = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2,
            )
        )

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        y = space_to_channel(x, self.stride)
        return match_channels(y, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for u in self.units:
            h = u(h)
        main = self.down(self.act(h))
        side = self._shortcut(x)
        if side.shape[-1] != main.shape[-1]:
            side = side[..., : main.shape[-1]]
        return main + side


class DecoderBlock(nn.Module):
    """Symmetric to :class:`EncoderBlock`: transpose-conv upsample
    followed by a stack of dilated residual units, with a non-parametric
    channel-to-space + channel-match shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations: tuple[int, ...] = (1, 3, 9),
        kernel: int = 7,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = Snake(in_channels)
        self.up = weight_norm(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2,
            )
        )
        self.units = nn.ModuleList(
            [DilatedResUnit(out_channels, kernel=kernel, dilation=d) for d in dilations]
        )

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if y.shape[1] % self.stride != 0:
            y = match_channels(y, (y.shape[1] // self.stride) * self.stride)
        y = channel_to_space(y, self.stride)
        return match_channels(y, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.up(self.act(x))
        side = self._shortcut(x)
        if side.shape[-1] < main.shape[-1]:
            main = main[..., : side.shape[-1]]
        elif side.shape[-1] > main.shape[-1]:
            side = side[..., : main.shape[-1]]
        h = main + side
        for u in self.units:
            h = u(h)
        return h
