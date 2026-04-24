"""Minimal YAML config loader with dotted access."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """dict subclass with attribute access (recursive)."""

    def __init__(self, data: Mapping[str, Any] | None = None):
        super().__init__()
        if data is None:
            return
        for k, v in data.items():
            self[k] = Config(v) if isinstance(v, Mapping) else v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return Config(raw or {})
