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
