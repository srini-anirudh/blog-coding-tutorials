"""Token windows from C5-format shards (uint16 .bin files listed in meta.json), held on the training device.

Every run reads windows from the start of the train stream, so a run of D tokens sees the first D tokens once.
A run with `unique` < D sees the first `unique` tokens again and again, reshuffled each epoch (section 9).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data_cache" / "tinystories"


def load_tokens(data_dir: Path, split: str, limit: int | None = None) -> np.ndarray:
    meta = json.loads((Path(data_dir) / "meta.json").read_text())
    parts, have = [], 0
    for s in meta[split]["shards"]:
        if limit is not None and have >= limit:
            break
        arr = np.memmap(Path(data_dir) / s["file"], dtype=np.uint16, mode="r")
        take = arr if limit is None else arr[: limit - have]
        parts.append(np.asarray(take))
        have += take.size
    return np.concatenate(parts)


class Windows:
    """Fixed-length windows of seq+1 tokens (inputs and shifted labels), stride seq."""

    def __init__(self, tokens: np.ndarray, seq: int, device: torch.device):
        self.seq = seq
        self.tokens = torch.from_numpy(tokens.astype(np.int64)).to(device)
        self.count = (len(tokens) - 1) // seq
        self.offsets = torch.arange(seq + 1, device=device)

    def get(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.tokens[idx[:, None] * self.seq + self.offsets]
        return w[:, :-1], w[:, 1:]


def epoch_order(n_windows: int, total: int, seed: int, device: torch.device) -> torch.Tensor:
    """Indices of `total` windows drawn from the first n_windows: a fresh permutation every epoch."""
    g = torch.Generator().manual_seed(seed)
    chunks, have = [], 0
    while have < total:
        chunks.append(torch.randperm(n_windows, generator=g))
        have += n_windows
    return torch.cat(chunks)[:total].to(device)
