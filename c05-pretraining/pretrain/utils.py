"""Device selection, seeding and memory readings that work on CUDA, MPS and CPU."""

from __future__ import annotations

import os
import random
import resource
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]  # the c05-pretraining folder


def resolve(path: str | Path) -> Path:
    """Relative paths in configs are relative to this tutorial folder, not the cwd."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def pick_device(name: str = "auto") -> torch.device:
    """cuda > mps > cpu, unless a device is named explicitly."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators


def synchronize(device: torch.device) -> None:
    """Wait for queued kernels. Timings without this measure launch time, not work."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_rss_mb() -> float:
    """Highest resident memory this process has reached, in MiB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux


def memory_mb(device: torch.device) -> float:
    """Memory currently in use, in MiB.

    CUDA and MPS: allocated tensor memory. CPU: the process's resident memory, because CPU
    tensors live in ordinary process memory. macOS has no /proc, so there it is the peak.
    """
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    return peak_rss_mb()
