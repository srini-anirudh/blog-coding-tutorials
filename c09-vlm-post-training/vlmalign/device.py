"""Device and seeding helpers shared by every script."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return cuda > mps > cpu, unless `prefer` (or $VLMALIGN_DEVICE) names one."""
    prefer = prefer or os.environ.get("VLMALIGN_DEVICE")
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_threads() -> None:
    """Honour $OMP_NUM_THREADS, else use at most 8 threads. Export a lower number on a shared machine."""
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", min(8, os.cpu_count() or 1))))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync(device: torch.device) -> None:
    """Wait for queued kernels so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
