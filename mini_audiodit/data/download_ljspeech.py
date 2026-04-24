"""Optional helper to fetch LJSpeech.  Run::

    python -m mini_audiodit.data.download_ljspeech data/raw
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "LJSpeech-1.1.tar.bz2"
    extracted = root / "LJSpeech-1.1"
    if extracted.exists():
        print(f"Already extracted at {extracted}.")
        return
    if not archive.exists():
        print(f"Downloading LJSpeech from {URL} ...")
        with urllib.request.urlopen(URL) as r, archive.open("wb") as f:
            shutil.copyfileobj(r, f)
    print(f"Extracting {archive} ...")
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(root)
    print(f"Done.  Point datasets at {extracted}.")


if __name__ == "__main__":
    sys.exit(main())
