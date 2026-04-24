"""Generic Euler ODE solver for flow matching.

Implements eq. 6 of the paper (minus the prompt / guidance machinery --
those live in :mod:`mini_audiodit.solvers.euler_tts`).
"""
from __future__ import annotations

from typing import Callable

import torch


VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class EulerSolver:
    """Explicit Euler integration from t=0 (noise) to t=1 (data).

    Works for any ``v(z_t, t)`` obeying the paper's convention
    ``dz_t/dt = v``.
    """

    def __init__(self, num_steps: int = 16) -> None:
        self.num_steps = num_steps

    @torch.no_grad()
    def integrate(
        self,
        velocity_fn: VelocityFn,
        z0: torch.Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        dt = (t_end - t_start) / self.num_steps
        z = z0
        traj: list[torch.Tensor] = [z.clone()] if return_trajectory else []
        for i in range(self.num_steps):
            t = torch.full((z.shape[0],), t_start + i * dt, device=z.device, dtype=z.dtype)
            v = velocity_fn(z, t)
            z = z + v * dt
            if return_trajectory:
                traj.append(z.clone())
        return traj if return_trajectory else z
