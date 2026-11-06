"""Device, seed and thread helpers shared by every script."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return cuda > mps > cpu, unless `prefer` (or $VLMTRAIN_DEVICE) names one."""
    prefer = prefer or os.environ.get("VLMTRAIN_DEVICE")
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def laptop_threads(n: int = 2) -> None:
    """Many tutorials share one laptop, so every run caps its CPU threads."""
    torch.set_num_threads(n)


def sync(device: torch.device) -> None:
    """Wait for queued kernels so wall-clock timings mean something."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_mb(device: torch.device) -> float:
    """Memory currently held by the allocator on `device`, in MB (0 on CPU)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return 0.0
