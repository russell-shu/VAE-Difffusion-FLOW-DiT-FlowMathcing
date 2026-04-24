"""TTS-specific Euler solver -- stage 10.

Implements eq. 6/7 of the LongCat-AudioDiT paper:

* per-step Euler update `z_{t+dt} = z_t + v * dt`
* **mismatch fix**: overwrite the prompt segment
  `z_t^ctx <- t z_ctx + (1-t) z_0^ctx` at every step (eq. 7)
* optional CFG / APG post-processing of the velocity

Guidance callables receive ``(v_cond, v_uncond, z_t, t)`` and return an
adjusted velocity.
"""
from __future__ import annotations

from typing import Callable, Protocol

import torch


class VelocityFn(Protocol):
    def __call__(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_ctx: torch.Tensor | None,
        cond: dict | None,
    ) -> torch.Tensor:
        ...


GuidanceFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class EulerTTSSolver:
    """Guidance-aware Euler solver with optional training-inference mismatch fix.

    Conceptual model:

    * The latent ``z_t`` is conceptually ``concat(z_t^ctx, z_t^gen)`` along the
      time axis.  ``mask`` is 1 on the **generation** region and 0 on the
      **prompt** region (matching ``m`` in paper eq. 4).
    * At every step we predict one conditional velocity using
      ``(z_ctx, cond)`` and (for CFG / APG) one unconditional velocity with
      ``z_ctx`` and the conditioning both dropped.
    * If ``fix_mismatch`` is True, we overwrite ``z_t`` on the prompt region
      with its ground-truth linear interpolation ``t * z_ctx + (1-t) * z0_ctx``
      *before* predicting the velocity on the next step (eq. 7).
    * Unconditional: per the corollary at the end of section 4.3, we also
      drop the noisy prompt ``z_t^ctx`` when building the unconditional
      input.  That is implemented by concatenating a zero-tensor on the
      prompt region of ``z_t`` for the unconditional branch.
    """

    def __init__(
        self,
        num_steps: int = 16,
        fix_mismatch: bool = True,
        drop_noisy_prompt_in_uncond: bool = True,
    ) -> None:
        self.num_steps = num_steps
        self.fix_mismatch = fix_mismatch
        self.drop_noisy_prompt_in_uncond = drop_noisy_prompt_in_uncond

    @torch.no_grad()
    def integrate(
        self,
        velocity_fn: VelocityFn,
        z0: torch.Tensor,
        z_ctx: torch.Tensor | None,
        mask: torch.Tensor | None,
        cond: dict | None = None,
        guidance: GuidanceFn | None = None,
    ) -> torch.Tensor:
        """Integrate from noise (t=0) to data (t=1).

        Args:
            velocity_fn: ``v(z_t, t, z_ctx, cond) -> v``.  Must handle
                ``z_ctx=None`` and ``cond=None`` (unconditional branch).
            z0: initial Gaussian noise of shape [B, C, T] (or similar).
            z_ctx: clean prompt latent z_ctx (same shape as z0, 0 outside prompt).
                   May be ``None`` for unconditional generation.
            mask:  1 on generation region, 0 on prompt region (broadcastable to z0).
                   May be ``None`` for unconditional.
            cond:  additional conditioning dict (e.g. ``{"text": q}``).
            guidance: callable applied to ``(v_cond, v_uncond, z_t, t)``.
        """
        dt = 1.0 / self.num_steps
        z = z0.clone()
        z0_ctx = z0.clone()
        if mask is not None:
            z0_ctx = z0_ctx * (1.0 - mask)

        for i in range(self.num_steps):
            t_val = i * dt
            t = torch.full((z.shape[0],), t_val, device=z.device, dtype=z.dtype)

            v_cond = velocity_fn(z, t, z_ctx, cond)
            if guidance is not None:
                z_u = z
                if self.drop_noisy_prompt_in_uncond and mask is not None:
                    z_u = z * mask
                v_uncond = velocity_fn(z_u, t, None, None)
                v = guidance(v_cond, v_uncond, z, t)
            else:
                v = v_cond

            z = z + v * dt

            if self.fix_mismatch and z_ctx is not None and mask is not None:
                t_next = min(t_val + dt, 1.0)
                gt_ctx = t_next * z_ctx + (1.0 - t_next) * z0_ctx
                z = mask * z + (1.0 - mask) * gt_ctx

        return z
