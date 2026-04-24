"""VoiceBox-style span masking utilities (LongCat-AudioDiT eq. 4)."""
from __future__ import annotations

import torch


def random_span_mask(
    length: int,
    batch: int,
    device: torch.device,
    min_gen_frac: float = 0.3,
    max_gen_frac: float = 0.7,
) -> torch.Tensor:
    """Return ``m`` of shape ``[B, 1, T]`` with 1 on the *generation* span.

    The complement ``1 - m`` marks the *prompt/context* region that stays
    visible to the model via ``z_ctx = z1 * (1 - m)``.
    """
    masks = []
    for _ in range(batch):
        frac = torch.empty((), device=device).uniform_(min_gen_frac, max_gen_frac)
        gen_len = int(torch.clamp(frac * length, min=1, max=length).item())
        start = torch.randint(0, length - gen_len + 1, (1,), device=device).item()
        m = torch.zeros(length, device=device)
        m[start : start + gen_len] = 1.0
        masks.append(m)
    m = torch.stack(masks, dim=0).unsqueeze(1)
    return m


def apply_cfg_drop(
    q: torch.Tensor,
    q_pad: torch.Tensor,
    z_ctx: torch.Tensor,
    drop_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """With probability ``drop_prob`` zero out text + context for CFG training."""
    if drop_prob <= 0.0:
        return q, q_pad, z_ctx
    b = q.shape[0]
    drop = (torch.rand(b, device=q.device) < drop_prob)
    if not drop.any():
        return q, q_pad, z_ctx
    q2 = q.clone()
    z2 = z_ctx.clone()
    pad2 = q_pad.clone()
    q2[drop] = 0.0
    z2[drop] = 0.0
    pad2[drop] = True
    return q2, pad2, z2
