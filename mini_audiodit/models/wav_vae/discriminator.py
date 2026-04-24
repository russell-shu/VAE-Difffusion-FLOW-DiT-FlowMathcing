"""Multi-scale STFT discriminator (HiFi-GAN / SoundStream style).

Each scale runs an STFT, takes the log-magnitude, and applies a small
stack of 2D convolutions (treated as ``Conv2d`` on ``[B, 1, F, T]``).
We return a list of logits (one per scale) **and** intermediate feature
maps for feature-matching loss.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft: int, hop: int, win: int, channels: int = 32) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(7, 7), padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Conv2d(channels * 4, 1, kernel_size=(3, 3), padding=1)

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1) if x.dim() == 3 else x
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window.to(x.device),
            return_complex=True,
        )
        mag = spec.abs().clamp_min(1e-5).log()

        return mag.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        mag = self._stft(x)
        feats: list[torch.Tensor] = []
        h = mag
        for layer in self.net:
            h = layer(h)
            feats.append(h)
        logits = self.head(h)
        return logits, feats


class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hops: Sequence[int] = (128, 256, 512),
        wins: Sequence[int] = (512, 1024, 2048),
    ) -> None:
        super().__init__()
        assert len(fft_sizes) == len(hops) == len(wins)
        self.discs = nn.ModuleList(
            [STFTDiscriminator(n_fft=n, hop=h, win=w) for n, h, w in zip(fft_sizes, hops, wins)]
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        logits: list[torch.Tensor] = []
        feats: list[list[torch.Tensor]] = []
        for d in self.discs:
            logit, f = d(x)
            logits.append(logit)
            feats.append(f)
        return logits, feats
