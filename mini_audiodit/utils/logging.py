"""Tiny logger that writes both to stdout and to a run directory."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(name: str, run_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
