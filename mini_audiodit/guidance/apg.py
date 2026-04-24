"""Adaptive Projection Guidance -- LongCat-AudioDiT eq. 9-10.

Given a flow-matching network that predicts velocity ``v_t``:

1. Map cond/uncond velocities into the *data* (sample) domain::

       mu_t  = z_t + (1 - t) * v_t
       mu_u  = z_t + (1 - t) * v_u
       dmu_t = mu_t - mu_u                         # guidance residual

2. Decompose ``dmu_t`` into components parallel / orthogonal to ``mu_t``::

       dmu_par  =  <dmu, mu> / <mu, mu>  * mu
       dmu_perp =  dmu - dmu_par

3. Dampen the parallel component with ``eta`` (default 0.5)::

       mu_APG = mu_t + alpha * dmu_perp + eta * dmu_par

4. Project back to velocity::

       v_APG = (mu_APG - z_t) / (1 - t)

The paper also keeps a reverse-momentum moving average of ``dmu`` with
coefficient ``beta`` (default -0.3); that is implemented via
:class:`APGState`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class APGConfig:
    alpha: float = 4.0
    eta: float = 0.5
    beta: float = -0.3
    eps: float = 1e-8
    time_epsilon: float = 1e-4


class APGState:
    """Holds the running moving average of ``dmu`` across Euler steps."""

    def __init__(self, config: APGConfig) -> None:
        self.config = config
        self.prev_dmu: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.prev_dmu = None


def _project(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    """Component of ``a`` along ``b``, computed per sample."""
    dims = tuple(range(1, a.dim()))
    num = (a * b).sum(dim=dims, keepdim=True)
    den = (b * b).sum(dim=dims, keepdim=True).clamp_min(eps)
    return (num / den) * b


def apg_velocity(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    state: APGState,
) -> torch.Tensor:
    cfg = state.config
    t_view = t.view((-1,) + (1,) * (z_t.dim() - 1))
    one_minus_t = (1.0 - t_view).clamp_min(cfg.time_epsilon)

    mu_t = z_t + one_minus_t * v_cond
    mu_u = z_t + one_minus_t * v_uncond
    dmu = mu_t - mu_u

    if cfg.beta != 0.0 and state.prev_dmu is not None:
        dmu = dmu + cfg.beta * state.prev_dmu
    state.prev_dmu = dmu.detach()

    dmu_par = _project(dmu, mu_t, cfg.eps)
    dmu_perp = dmu - dmu_par

    mu_apg = mu_t + cfg.alpha * dmu_perp + cfg.eta * dmu_par
    v_apg = (mu_apg - z_t) / one_minus_t
    return v_apg
