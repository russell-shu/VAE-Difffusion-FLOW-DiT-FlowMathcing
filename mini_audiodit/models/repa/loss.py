"""Representation alignment loss (Yu 2024; LongCat-AudioDiT section 4.1)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class REPALoss(nn.Module):
    """L1 between a DiT token sequence and a HuBERT sequence.

    Both sequences are length-normalised with adaptive average pooling on
    the time axis before comparison.
    """

    def __init__(self, dit_dim: int, hub_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dit_dim, hub_dim)

    def forward(self, dit_tokens: torch.Tensor, hubert: torch.Tensor) -> torch.Tensor:
        """``dit_tokens``: [B, L, C_dit], ``hubert``: [B, T, C_hub] (already detached)."""
        h = self.proj(dit_tokens).transpose(1, 2)
        t = hubert.detach().transpose(1, 2)
        length = min(h.shape[-1], t.shape[-1])
        h = F.adaptive_avg_pool1d(h, length).transpose(1, 2)
        t = F.adaptive_avg_pool1d(t, length).transpose(1, 2)
        return (h - t).abs().mean()
