"""Raw corpus: download TinyStories (V2, GPT-4 generated) and split it into documents.

The files are plain text with stories separated by the literal string "<|endoftext|>".
The train file is 2.2 GB, so we fetch only its first `train_bytes` with an HTTP Range
request. That is enough text for a ~10 minute laptop run.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}
SEPARATOR = "<|endoftext|>"


def download(split: str, out_dir: Path, max_bytes: int | None = None) -> Path:
    """Fetch one split (optionally only its first max_bytes) into out_dir. Cached."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FILES[split]
    if path.exists():
        return path
    req = urllib.request.Request(f"{BASE}/{FILES[split]}")
    if max_bytes is not None:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


def split_documents(text: str, drop_last_partial: bool = False) -> list[str]:
    """Split raw text into stripped, non-empty documents.

    A Range download cuts the final story mid-sentence; drop_last_partial removes it
    so no truncated document enters the corpus.
    """
    parts = text.split(SEPARATOR)
    if drop_last_partial and len(parts) > 1:
        parts = parts[:-1]
    return [p.strip() for p in parts if p.strip()]


def load_documents(split: str, data_dir: Path, train_bytes: int = 128 * 2**20) -> list[str]:
    max_bytes = train_bytes if split == "train" else None
    path = download(split, data_dir / "raw", max_bytes)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return split_documents(text, drop_last_partial=max_bytes is not None)
