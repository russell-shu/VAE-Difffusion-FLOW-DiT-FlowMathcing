"""VoiceBox-style span masking used by paper eq. 4.

A single random span per sample is mapped to ``m = 1``; outside the
span ``m = 0``.  Convention matches the paper:

* ``m_i = 1``  -- *prediction* region (the DiT has to fill this in).
* ``m_i = 0``  -- *prompt*     region (ground-truth latent is provided).

``z_ctx = (1 - m) * z_1`` is the prompt latent (zeroed out on the
prediction span).  The masked CFM loss is therefore

    L_CFM = E [ || m ⊙ ( (z_1 - z_0) - v_theta(z_t, t, z_ctx, q) ) ||^2 ]

i.e. the loss is computed *only* on the prediction span.
"""
from __future__ import annotations

import torch


def random_span_mask(
    batch_size: int,
    seq_len: int,
    min_ratio: float = 0.7,
    max_ratio: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample a per-item mask with a single *contiguous* prediction span.

    Returns ``[B, 1, L]`` with ``1`` in the masked (predict) region.

    The ratio (mask length / seq_len) is drawn uniformly from
    ``[min_ratio, max_ratio]`` -- VoiceBox's original recipe.  Setting
    both to ``1.0`` trivialises the model to fully-generative TTS.
    """
    device = device or torch.device("cpu")
    mask = torch.zeros(batch_size, 1, seq_len, device=device)
    for b in range(batch_size):
        ratio = float(torch.empty(1).uniform_(min_ratio, max_ratio).item())
        span = max(1, int(ratio * seq_len))
        start = int(torch.randint(0, seq_len - span + 1, (1,)).item()) if seq_len > span else 0
        mask[b, 0, start : start + span] = 1.0
    return mask


def build_tts_inputs(
    z1: torch.Tensor,
    min_ratio: float = 0.7,
    max_ratio: float = 1.0,
    drop_ctx_prob: float = 0.1,
    drop_text_prob: float = 0.1,
    device: torch.device | None = None,
) -> dict:
    """Produce everything needed for a masked-CFM training step.

    Returns a dict with:

    * ``z1``      -- original clean latent.
    * ``z0``      -- prior noise of same shape.
    * ``z_t``     -- linear interpolant at sampled ``t``.
    * ``t``       -- [B] timesteps in ``(0, 1)``.
    * ``mask``    -- [B, 1, L] float mask (1 = predict, 0 = prompt).
    * ``z_ctx``   -- ``(1 - mask) * z1`` (possibly all-zeros after drop).
    * ``drop_ctx`` -- [B] bool, True when we **dropped** the prompt for CFG.
    * ``drop_text`` -- [B] bool, True when we **dropped** text for CFG.
    * ``v_target``-- target velocity ``z1 - z0``.
    """
    device = device or z1.device
    B, C, L = z1.shape
    mask = random_span_mask(B, L, min_ratio=min_ratio, max_ratio=max_ratio, device=device)

    z0 = torch.randn_like(z1)
    t = torch.rand(B, device=device).clamp(1e-5, 1.0 - 1e-5)
    t_view = t.view(-1, 1, 1)
    z_t = (1.0 - t_view) * z0 + t_view * z1

    drop_ctx = torch.rand(B, device=device) < drop_ctx_prob
    drop_text = torch.rand(B, device=device) < drop_text_prob

    z_ctx_clean = (1.0 - mask) * z1
    z_ctx = torch.where(drop_ctx.view(-1, 1, 1), torch.zeros_like(z_ctx_clean), z_ctx_clean)

    v_target = z1 - z0
    return {
        "z1": z1,
        "z0": z0,
        "z_t": z_t,
        "t": t,
        "mask": mask,
        "z_ctx": z_ctx,
        "drop_ctx": drop_ctx,
        "drop_text": drop_text,
        "v_target": v_target,
    }


def concat_dit_input(z_t: torch.Tensor, z_ctx: torch.Tensor | None) -> torch.Tensor:
    """Concatenate ``z_t`` and ``z_ctx`` along the channel dim.

    The paper feeds the DiT a ``[z_t || z_ctx]`` input of size ``2 C``.
    For the unconditional branch you can pass ``z_ctx=None`` and we'll
    concatenate a zero tensor of the same shape (paper's CFG
    corollary).
    """
    if z_ctx is None:
        z_ctx = torch.zeros_like(z_t)
    return torch.cat([z_t, z_ctx], dim=1)
