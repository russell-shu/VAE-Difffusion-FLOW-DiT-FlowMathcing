"""Tiny 2D UNet used by stages 2-4 (MNIST).

Implements sinusoidal time embedding, optional label embedding, two
down-/up-sampling stages, and a 3-layer bottleneck.  Predicting ``eps``
(DDPM), ``v`` (flow matching), or ``x0`` is simply a matter of how the
output is used -- the network architecture is identical.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal embedding.  ``t`` has shape ``[B]`` or ``[B,1]``."""
    if t.dim() == 2:
        t = t.squeeze(-1)
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeCondBlock(nn.Module):
    """Residual conv block with a FiLM-style time/class shift."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.cond = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.cond(c).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet2D(nn.Module):
    """Tiny UNet with two down stages and two up stages."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        time_dim: int = 128,
        num_classes: int | None = None,
        class_drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.num_classes = num_classes
        self.class_drop_prob = class_drop_prob

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(), nn.Linear(time_dim * 2, time_dim),
        )
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes + 1, time_dim)
            self.null_class = num_classes
        else:
            self.class_emb = None

        c = base_channels
        self.stem = nn.Conv2d(in_channels, c, 3, padding=1)
        self.d1 = TimeCondBlock(c, c, time_dim)
        self.d2 = TimeCondBlock(c, c * 2, time_dim)
        self.d3 = TimeCondBlock(c * 2, c * 2, time_dim)
        self.mid1 = TimeCondBlock(c * 2, c * 4, time_dim)
        self.mid2 = TimeCondBlock(c * 4, c * 2, time_dim)
        self.u3 = TimeCondBlock(c * 4, c * 2, time_dim)
        self.u2 = TimeCondBlock(c * 4, c, time_dim)
        self.u1 = TimeCondBlock(c * 2, c, time_dim)
        self.out = nn.Conv2d(c, out_channels, 3, padding=1)

    def _cond(self, t: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        if self.class_emb is not None:
            if y is None:
                y = torch.full((t.shape[0],), self.null_class, dtype=torch.long, device=t.device)
            elif self.training and self.class_drop_prob > 0.0:
                drop = torch.rand(y.shape[0], device=y.device) < self.class_drop_prob
                y = torch.where(drop, torch.full_like(y, self.null_class), y)
            emb = emb + self.class_emb(y)
        return emb

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        c = self._cond(t, y)
        h0 = self.stem(x)
        h1 = self.d1(h0, c)
        h2 = self.d2(F.avg_pool2d(h1, 2), c)
        h3 = self.d3(F.avg_pool2d(h2, 2), c)
        m = self.mid1(h3, c)
        m = self.mid2(m, c)
        u3 = self.u3(torch.cat([m, h3], dim=1), c)
        u3 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = self.u2(torch.cat([u3, h2], dim=1), c)
        u2 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.u1(torch.cat([u2, h1], dim=1), c)
        return self.out(u1)
