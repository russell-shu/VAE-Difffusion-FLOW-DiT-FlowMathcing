"""Wav-VAE encoder / decoder.  Stage 5 (AE) and stage 6 (full VAE)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .blocks import EncoderBlock, DecoderBlock, Snake, match_channels


class WavVAEEncoder(nn.Module):
    """Encoder described in paper section 3.1."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (2, 4, 4, 8),
        strides: Sequence[int] = (2, 4, 4, 5),
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        assert len(channel_mults) == len(strides)
        self.latent_dim = latent_dim
        self.strides = tuple(strides)
        self.downsample_ratio = 1
        for s in strides:
            self.downsample_ratio *= s

        self.stem = weight_norm(nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3))
        blocks = []
        c_in = base_channels
        for mult, stride in zip(channel_mults, strides):
            c_out = base_channels * mult
            blocks.append(EncoderBlock(c_in, c_out, stride=stride))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.feature_channels = c_in

        self.proj_act = Snake(c_in)
        self.proj = weight_norm(nn.Conv1d(c_in, latent_dim, kernel_size=7, padding=3))

    def _proj_shortcut(self, x: torch.Tensor) -> torch.Tensor:
        return match_channels(x, self.latent_dim)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """``[B, 1, T] -> [B, latent_dim, T // downsample_ratio]``.

        In stage-5 ``WavAE`` this output is the deterministic latent.
        In stage-6 ``WavVAE`` it is split into ``mu / logvar``.
        """
        h = self.stem(wave)
        for blk in self.blocks:
            h = blk(h)
        main = self.proj(self.proj_act(h))
        side = self._proj_shortcut(h)
        if side.shape[-1] != main.shape[-1]:
            side = side[..., : main.shape[-1]]
        return main + side


class WavVAEDecoder(nn.Module):
    """Decoder mirrors the encoder in reverse."""

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (2, 4, 4, 8),
        strides: Sequence[int] = (2, 4, 4, 5),
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        assert len(channel_mults) == len(strides)
        self.latent_dim = latent_dim
        self.strides = tuple(strides)
        self.upsample_ratio = 1
        for s in strides:
            self.upsample_ratio *= s

        feature_channels = base_channels * channel_mults[-1]
        self.stem = weight_norm(
            nn.Conv1d(latent_dim, feature_channels, kernel_size=7, padding=3)
        )
        self.feature_channels = feature_channels

        blocks = []
        c_in = feature_channels
        reversed_mults = list(channel_mults[::-1])
        reversed_strides = list(strides[::-1])
        for i, (mult, stride) in enumerate(zip(reversed_mults, reversed_strides)):
            c_out = base_channels * reversed_mults[i + 1] if i + 1 < len(reversed_mults) else base_channels
            blocks.append(DecoderBlock(c_in, c_out, stride=stride))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)

        self.out_act = Snake(c_in)
        self.out = weight_norm(nn.Conv1d(c_in, out_channels, kernel_size=7, padding=3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)
        for blk in self.blocks:
            h = blk(h)
        return self.out(self.out_act(h))


class WavAE(nn.Module):
    """Stage-5 model: deterministic autoencoder.  No KL, no GAN.

    Used as a training-stable baseline before stage 6 piles on the rest
    of the Wav-VAE objective.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (2, 4, 4, 8),
        strides: Sequence[int] = (2, 4, 4, 5),
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = WavVAEEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            strides=strides,
            latent_dim=latent_dim,
        )
        self.decoder = WavVAEDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            strides=strides,
            latent_dim=latent_dim,
        )
        self.downsample_ratio = self.encoder.downsample_ratio

    def encode(self, wave: torch.Tensor) -> torch.Tensor:
        return self.encoder(wave)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(wave)
        rec = self.decode(z)
        if rec.shape[-1] != wave.shape[-1]:
            rec = rec[..., : wave.shape[-1]]
        return rec, z


class WavVAE(nn.Module):
    """Full Wav-VAE with a Gaussian bottleneck.

    Encoder produces 2 * D channels (mu and logvar), stage-6 training
    samples ``z = mu + sigma * eps`` and decodes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (2, 4, 4, 8),
        strides: Sequence[int] = (2, 4, 4, 5),
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = WavVAEEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            strides=strides,
            latent_dim=latent_dim * 2,
        )
        self.decoder = WavVAEDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            strides=strides,
            latent_dim=latent_dim,
        )
        self.downsample_ratio = self.encoder.downsample_ratio

    def encode(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(wave)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, wave: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(wave)
        z = self.reparameterize(mu, logvar)
        rec = self.decode(z)
        if rec.shape[-1] != wave.shape[-1]:
            rec = rec[..., : wave.shape[-1]]
        return rec, z, mu, logvar
