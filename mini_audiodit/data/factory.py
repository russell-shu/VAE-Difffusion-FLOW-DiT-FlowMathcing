"""Dataset factory used by stages 5-12."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from .audio import LJSpeechDataset, SyntheticAudioDataset, TextAudioDataset


def get_audio_dataset(
    name: str,
    root: str | Path,
    sample_rate: int,
    segment_seconds: float,
    max_items: int | None,
    synthetic_items: int,
    return_text: bool = False,
) -> Dataset:
    name = name.lower()
    if name == "synthetic":
        return SyntheticAudioDataset(
            num_items=synthetic_items,
            sample_rate=sample_rate,
            duration=segment_seconds,
        )
    if name == "ljspeech":
        return LJSpeechDataset(
            root=root,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            max_items=max_items,
            return_text=return_text,
        )
    raise ValueError(f"unknown dataset {name!r} (expected 'synthetic' or 'ljspeech')")


def wrap_with_text(ds: Dataset, texts: list[str] | None) -> Dataset:
    if texts is None:
        return ds
    return TextAudioDataset(ds, texts)
