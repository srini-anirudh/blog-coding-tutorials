"""Small datasets: two interleaved spirals, a 1-D target function, and a byte-level text corpus."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch


def two_spirals(n: int = 2000, noise: float = 0.05, turns: float = 1.5, seed: int = 0):
    """n points on two interleaved spirals in 2-D, labels 0 and 1. No straight line separates them."""
    gen = torch.Generator().manual_seed(seed)
    half = n // 2
    t = torch.rand(half, generator=gen) * turns * 2 * math.pi + 0.5
    arm = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1) / (turns * 2 * math.pi)
    x = torch.cat([arm, -arm]) + noise * torch.randn(2 * half, 2, generator=gen)
    y = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])
    perm = torch.randperm(2 * half, generator=gen)
    return x[perm], y[perm]


def wikitext_parquet_dir() -> Path | None:
    """The wikitext-2-raw-v1 folder in the local Hugging Face cache, if it was downloaded already."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snaps = home / "hub" / "datasets--Salesforce--wikitext" / "snapshots"
    for d in sorted(snaps.glob("*/wikitext-2-raw-v1")):
        if (d / "train-00000-of-00001.parquet").exists():
            return d
    return None


def wikitext_bytes(split: str) -> torch.Tensor:
    """wikitext-2 as one long uint8 tensor of UTF-8 bytes. Uses the cache, or downloads with `datasets`."""
    d = wikitext_parquet_dir()
    if d is not None:
        import pyarrow.parquet as pq

        lines = pq.read_table(d / f"{split}-00000-of-00001.parquet").column("text").to_pylist()
    else:
        from datasets import load_dataset

        lines = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]
    data = "".join(lines).encode("utf-8")
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def synthetic_bytes(n: int, seed: int = 0) -> torch.Tensor:
    """Offline stand-in for tests: repeated short sentences with some structure to learn."""
    gen = torch.Generator().manual_seed(seed)
    words = ["the cat ", "a dog ", "sat on ", "the mat. ", "ran to ", "a tree. "]
    idx = torch.randint(len(words), (n // 6 + 1,), generator=gen).tolist()
    text = "".join(words[i] for i in idx)[:n]
    return torch.frombuffer(bytearray(text.encode()), dtype=torch.uint8).clone()


def random_windows(data: torch.Tensor, batch: int, seq: int, gen: torch.Generator) -> torch.Tensor:
    """A (batch, seq) LongTensor of random contiguous windows."""
    starts = torch.randint(len(data) - seq, (batch,), generator=gen)
    return torch.stack([data[s : s + seq] for s in starts.tolist()]).long()
