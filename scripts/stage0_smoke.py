"""Stage-0 smoke test.

Loads one MNIST batch, builds one synthetic audio batch, writes a waveform
and mel-spectrogram PNG to ``runs/stage0/``.  Should run in a few seconds
on CPU and does not require downloading anything besides MNIST.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_audiodit.data import (
    SyntheticAudioDataset,
    build_audio_loader,
    build_mnist_loaders,
)
from mini_audiodit.utils import save_mel_spectrogram, save_waveform


def main() -> None:
    run_dir = ROOT / "runs" / "stage0"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, _ = build_mnist_loaders(ROOT / "data/mnist", batch_size=32)
    x, y = next(iter(train_loader))
    print(f"MNIST batch: images {tuple(x.shape)}, labels {tuple(y.shape)}")

    ds = SyntheticAudioDataset(num_items=8, sample_rate=24000, duration=2.0)
    loader = build_audio_loader(ds, batch_size=4, shuffle=False)
    audio = next(iter(loader))
    print(f"Audio batch: {tuple(audio.shape)} at 24 kHz")

    save_waveform(audio[0, 0], run_dir / "wave_sample", sample_rate=24000)
    save_mel_spectrogram(audio[0, 0], run_dir / "mel_sample", sample_rate=24000)
    print(f"Wrote waveform + mel to {run_dir}")


if __name__ == "__main__":
    main()
