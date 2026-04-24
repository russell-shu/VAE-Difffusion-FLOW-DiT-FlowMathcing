"""Audio datasets and collates used by stages 5-12.

Three datasets are provided:

* :class:`SyntheticAudioDataset` -- deterministic sine/FM/noise mixtures.
  Useful for smoke tests, local CI, and when LJSpeech is not downloaded.
* :class:`LJSpeechDataset` -- the real dataset, single English speaker.
* :class:`TextAudioDataset` -- wraps an audio dataset and a parallel
  list of transcripts, used by stages 8+ (text-conditioned TTS).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class SyntheticAudioDataset(Dataset):
    """Procedural sine + FM + pink-noise mixtures.

    Guarantees every stage is runnable without real audio.  Each sample
    is deterministic given ``(index, seed)``.
    """

    def __init__(
        self,
        num_items: int = 200,
        sample_rate: int = 24000,
        duration: float = 3.0,
        seed: int = 0,
    ) -> None:
        self.num_items = num_items
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.seed = seed

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed + idx)
        t = torch.arange(self.num_samples, dtype=torch.float32) / self.sample_rate
        f0 = 110.0 * (2.0 ** (torch.randint(0, 24, (1,), generator=g).item() / 12.0))
        mod = 3.0 + 3.0 * torch.rand(1, generator=g).item()
        depth = 40.0 + 60.0 * torch.rand(1, generator=g).item()
        carrier = torch.sin(2 * math.pi * f0 * t + depth / mod * torch.sin(2 * math.pi * mod * t))
        noise = 0.02 * torch.randn(self.num_samples, generator=g)
        env = torch.sin(math.pi * t / self.duration).clamp_min(0)
        wave = (0.7 * carrier + noise) * env
        return wave.unsqueeze(0)


class LJSpeechDataset(Dataset):
    """LJSpeech audio-only dataset.

    Expects the standard LJSpeech-1.1 layout::

        <root>/wavs/LJ001-0001.wav
        <root>/metadata.csv
    """

    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 24000,
        segment_seconds: float | None = 3.0,
        max_items: int | None = None,
        return_text: bool = False,
    ) -> None:
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.return_text = return_text
        meta_path = self.root / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.csv not found at {meta_path}. "
                "Download LJSpeech or use SyntheticAudioDataset."
            )
        items: list[tuple[str, str]] = []
        with meta_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) < 2:
                    continue
                wav_id, text = row[0], row[-1]
                items.append((wav_id, text))
        if max_items is not None:
            items = items[:max_items]
        self.items = items
        self._resampler_cache: dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _load_wave(self, wav_id: str) -> torch.Tensor:
        path = self.root / "wavs" / f"{wav_id}.wav"
        wave, sr = torchaudio.load(str(path))
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            if sr not in self._resampler_cache:
                self._resampler_cache[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
            wave = self._resampler_cache[sr](wave)
        if self.segment_seconds is not None:
            n = int(self.segment_seconds * self.sample_rate)
            if wave.shape[-1] >= n:
                start = torch.randint(0, wave.shape[-1] - n + 1, (1,)).item()
                wave = wave[..., start : start + n]
            else:
                pad = n - wave.shape[-1]
                wave = torch.nn.functional.pad(wave, (0, pad))
        return wave

    def __getitem__(self, idx: int):
        wav_id, text = self.items[idx]
        wave = self._load_wave(wav_id)
        if self.return_text:
            return wave, text
        return wave


class TextAudioDataset(Dataset):
    """A wrapper that pairs audio with transcripts for stages 8+."""

    def __init__(self, audio_ds: Dataset, texts: Sequence[str] | None = None) -> None:
        self.audio_ds = audio_ds
        if texts is None:
            texts = [f"synthetic audio sample {i}" for i in range(len(audio_ds))]
        self.texts = list(texts)
        if len(self.texts) < len(audio_ds):
            repeats = math.ceil(len(audio_ds) / len(self.texts))
            self.texts = (self.texts * repeats)[: len(audio_ds)]

    def __len__(self) -> int:
        return len(self.audio_ds)

    def __getitem__(self, idx: int):
        item = self.audio_ds[idx]
        if isinstance(item, tuple):
            wave, text = item
        else:
            wave, text = item, self.texts[idx]
        return wave, text


def collate_audio(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack fixed-length waveforms into [B, 1, T]."""
    return torch.stack(list(batch), dim=0)


def collate_text_audio(batch):
    waves, texts = zip(*batch)
    return torch.stack(list(waves), dim=0), list(texts)


def build_audio_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    text: bool = False,
) -> DataLoader:
    collate_fn = collate_text_audio if text else collate_audio
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
