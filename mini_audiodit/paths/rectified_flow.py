"""Rectified-flow / CFM path (LongCat-AudioDiT eq. 3-4).

Paper convention (watch out, it is *reversed* vs many other flow papers):

    z_0 ~ N(0, I)      (noise)
    z_1 ~ data
    z_t = (1 - t) z_0 + t z_1,   t in [0, 1]

Therefore dz_t/dt = z_1 - z_0 is the **target velocity**.  Inference
integrates from t=0 (noise) to t=1 (data).
"""
from __future__ import annotations

import torch


class RectifiedFlow:
    """Thin stateless helper -- no learnable parameters."""

    @staticmethod
    def sample_t(batch_size: int, device: torch.device, eps: float = 1e-5) -> torch.Tensor:
        return torch.rand(batch_size, device=device).clamp(eps, 1.0 - eps)

    @staticmethod
    def interpolate(
        z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """z_t = (1 - t) z_0 + t z_1."""
        shape = (-1,) + (1,) * (z0.dim() - 1)
        t = t.view(shape)
        return (1.0 - t) * z0 + t * z1

    @staticmethod
    def target_velocity(z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        return z1 - z0

    @staticmethod
    def draw_prior(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device)
