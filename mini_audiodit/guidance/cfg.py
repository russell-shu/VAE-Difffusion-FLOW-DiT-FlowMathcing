"""Classifier-free guidance (Ho & Salimans 2021; LongCat-AudioDiT eq. 8)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CFGConfig:
    alpha: float = 4.0


def cfg_velocity(
    v_cond: torch.Tensor, v_uncond: torch.Tensor, alpha: float = 4.0
) -> torch.Tensor:
    """v_CFG = v_cond + alpha * (v_cond - v_uncond).

    (Paper eq. 8; note the paper writes ``v_t + alpha (v_t - v_t^u)``.)
    """
    return v_cond + alpha * (v_cond - v_uncond)
