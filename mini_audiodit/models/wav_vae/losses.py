"""Auxiliary losses for the Wav-VAE objective (paper eq. 2)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MultiResSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (Zeghidour 2021 / SoundStream)."""

    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hops: Sequence[int] = (128, 256, 512),
        wins: Sequence[int] = (512, 1024, 2048),
    ) -> None:
        super().__init__()
        assert len(fft_sizes) == len(hops) == len(wins)
        self.specs = list(zip(fft_sizes, hops, wins))
        self.windows: dict[int, torch.Tensor] = {}

    def _window(self, n: int, device: torch.device) -> torch.Tensor:
        key = (n, str(device))
        if key not in self.windows:
            self.windows[key] = torch.hann_window(n, device=device)
        return self.windows[key]

    def _stft(self, x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        w = self._window(win, x.device)
        x = x.squeeze(1) if x.dim() == 3 else x
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=w, return_complex=True)
        return spec

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = pred.new_zeros(())
        for n_fft, hop, win in self.specs:
            P = self._stft(pred, n_fft, hop, win)
            T = self._stft(target, n_fft, hop, win)
            Pm = P.abs().clamp_min(1e-8)
            Tm = T.abs().clamp_min(1e-8)
            log_l1 = (Pm.log() - Tm.log()).abs().mean()
            sc = (Pm - Tm).pow(2).sum().sqrt() / Tm.pow(2).sum().sqrt().clamp_min(1e-8)
            loss = loss + log_l1 + sc
        return loss / len(self.specs)


class MultiScaleMelLoss(nn.Module):
    """L1 on mel magnitudes at several FFT sizes (Kumar 2023 / DAC)."""

    def __init__(
        self,
        sample_rate: int = 24000,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hops: Sequence[int] = (128, 256, 512),
        n_mels: Sequence[int] = (40, 80, 128),
    ) -> None:
        super().__init__()
        assert len(fft_sizes) == len(hops) == len(n_mels)
        self.mels = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop,
                    n_mels=n_m,
                    power=1.0,
                )
                for n_fft, hop, n_m in zip(fft_sizes, hops, n_mels)
            ]
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target
        loss = pred.new_zeros(())
        for mel in self.mels:
            pm = mel(pred).clamp_min(1e-5)
            tm = mel(target).clamp_min(1e-5)
            loss = loss + (pm - tm).abs().mean() + (pm.log() - tm.log()).abs().mean()
        return loss / len(self.mels)


class FeatureMatchingLoss(nn.Module):
    """HiFi-GAN style L1 between the discriminator's feature maps."""

    def forward(
        self, pred_feats: list[list[torch.Tensor]], target_feats: list[list[torch.Tensor]]
    ) -> torch.Tensor:
        loss = pred_feats[0][0].new_zeros(())
        n = 0
        for p_layers, t_layers in zip(pred_feats, target_feats):
            for p, t in zip(p_layers, t_layers):
                loss = loss + (p - t.detach()).abs().mean()
                n += 1
        return loss / max(n, 1)


def kl_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL( N(mu, sigma^2) || N(0, 1) ) averaged over batch, summed over features."""
    logvar = logvar.clamp(-10, 10)
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()
