"""Cheap objective metrics for stage-12 sanity checks.

The paper reports Whisper ASR WER, WavLM SIM, UTMOS, DNSMOS.  Those
require heavy extra dependencies / model weights.  This module keeps a
**unified interface** but falls back to lightweight surrogates when the
fancy toolkits are not installed:

* ``stft_l1`` -- multi-resolution log-magnitude L1 (always available)
* ``wave_l1`` -- time-domain L1

If you install ``openai-whisper``, ``stoi``, ``pesq``, etc., extend this
file -- the training script already calls :func:`evaluate_sample`.
"""
from __future__ import annotations

from typing import Any

import torch

from mini_audiodit.models.wav_vae import MultiResSTFTLoss


_stft = MultiResSTFTLoss()


@torch.no_grad()
def evaluate_sample(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    pred = pred.detach()
    ref = ref.detach()
    if pred.shape != ref.shape:
        m = min(pred.shape[-1], ref.shape[-1])
        pred = pred[..., :m]
        ref = ref[..., :m]
    out: dict[str, Any] = {}
    out["wave_l1"] = float((pred - ref).abs().mean().cpu())
    out["stft_l1"] = float(_stft(pred, ref).cpu())

    try:
        import pesq  # type: ignore

        p = pred[0, 0].cpu().numpy()
        r = ref[0, 0].cpu().numpy()
        out["pesq"] = float(pesq.pesq(16000, r, p, "wb"))
    except Exception:
        out["pesq"] = None

    try:
        from pystoi import stoi  # type: ignore

        p = pred[0, 0].cpu().numpy()
        r = ref[0, 0].cpu().numpy()
        out["stoi"] = float(stoi(r, p, 16000, extended=False))
    except Exception:
        out["stoi"] = None

    return out
