"""Minimal DDPM schedule + forward/backward helpers (stage 2)."""
from __future__ import annotations

import math

import torch


class DDPMSchedule:
    """Discrete-time DDPM with ``num_steps`` noise levels.

    Supports ``linear`` and ``cosine`` beta schedules (Ho 2020 / Nichol 2021).
    The model is trained to predict ``eps`` (the noise).
    """

    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, schedule: str = "linear") -> None:
        self.num_steps = num_steps
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
        elif schedule == "cosine":
            s = 0.008
            ts = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
            f = torch.cos(((ts + s) / (1 + s)) * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            betas = torch.clamp(1.0 - (alpha_bar[1:] / alpha_bar[:-1]), 1e-8, 0.999)
        else:
            raise ValueError(f"unknown schedule {schedule!r}")
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alpha_bar = alpha_bar.float()
        self.sqrt_alpha_bar = self.alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - self.alpha_bar).sqrt()

    def to(self, device: torch.device) -> "DDPMSchedule":
        for name in ["betas", "alphas", "alpha_bar", "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar"]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """q(x_t | x_0) = N(sqrt(a_bar) x_0, (1-a_bar) I), closed-form."""
        if noise is None:
            noise = torch.randn_like(x0)
        shape = (-1,) + (1,) * (x0.dim() - 1)
        a = self.sqrt_alpha_bar[t].view(shape)
        b = self.sqrt_one_minus_alpha_bar[t].view(shape)
        x_t = a * x0 + b * noise
        return x_t, noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        shape = (-1,) + (1,) * (x_t.dim() - 1)
        a = self.sqrt_alpha_bar[t].view(shape)
        b = self.sqrt_one_minus_alpha_bar[t].view(shape)
        return (x_t - b * eps) / a

    @torch.no_grad()
    def ancestral_sample(
        self, model, shape: tuple[int, ...], device: torch.device, num_steps: int | None = None,
    ) -> torch.Tensor:
        """Standard DDPM ancestral sampling (``p_theta(x_{t-1}|x_t)``)."""
        n = num_steps or self.num_steps
        step_indices = torch.linspace(self.num_steps - 1, 0, n, dtype=torch.long, device=device)
        x = torch.randn(shape, device=device)
        for i, t in enumerate(step_indices):
            t_batch = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
            eps = model(x, t_batch.float())
            view = (-1,) + (1,) * (x.dim() - 1)
            alpha_bar_t = self.alpha_bar[t].view(view)
            alpha_t = self.alphas[t].view(view)
            beta_t = self.betas[t].view(view)
            mean = (1.0 / alpha_t.sqrt()) * (x - (beta_t / (1.0 - alpha_bar_t).sqrt()) * eps)
            if i < len(step_indices) - 1:
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean
        return x
