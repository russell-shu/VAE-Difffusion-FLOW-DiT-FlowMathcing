"""Visualization helpers: image grids, waveforms, mel-spectrograms, histograms."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torchvision.utils import make_grid, save_image


def save_image_grid(tensor: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(tensor.detach().cpu().clamp(0, 1), nrow=nrow, padding=2)
    save_image(grid, path)


def save_waveform(wave: torch.Tensor, path: str | Path, sample_rate: int) -> None:
    """Save a 1D waveform to a PNG with a time axis and to a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wave = wave.detach().cpu().float()
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    torchaudio.save(str(path.with_suffix(".wav")), wave, sample_rate)

    fig, ax = plt.subplots(figsize=(8, 2))
    t = np.arange(wave.shape[-1]) / sample_rate
    ax.plot(t, wave[0].numpy(), linewidth=0.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amp")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)


def save_mel_spectrogram(
    wave: torch.Tensor,
    path: str | Path,
    sample_rate: int,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 80,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wave = wave.detach().cpu().float()
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0,
    )
    mel = mel_tf(wave).clamp_min(1e-8).log10()[0].numpy()
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(mel, origin="lower", aspect="auto")
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)


def plot_latent_histogram(z: torch.Tensor, path: str | Path, bins: int = 60) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z = z.detach().cpu().float().flatten().numpy()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(z, bins=bins, density=True, alpha=0.75)
    xs = np.linspace(-4, 4, 400)
    ax.plot(xs, np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi), label="N(0,1)")
    ax.set_title(f"latent histogram (mean={z.mean():.2f}, std={z.std():.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)
