"""Device, seed and thread helpers shared by every script."""

from __future__ import annotations

import os
import platform
import random
import resource
from pathlib import Path

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


def cap_threads(n: int | None = None) -> int:
    """Cap CPU threads: `n`, else $OMP_NUM_THREADS, else at most 8. Large machines are usually shared."""
    n = n or int(os.environ.get("OMP_NUM_THREADS", min(8, os.cpu_count() or 1)))
    torch.set_num_threads(n)
    return n


def describe_hardware(device: torch.device) -> str:
    """The label every number in results/ carries, for example "AMD EPYC 7R13 Processor, 8 threads"."""
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(device)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    if device.type == "mps":
        return "Apple MPS"
    model = next((line.split(":", 1)[1].strip() for line in _cpuinfo() if line.startswith("model name")),
                 platform.processor() or "CPU")
    return f"{model}, {torch.get_num_threads()} threads"


def _cpuinfo() -> list[str]:
    path = Path("/proc/cpuinfo")
    return path.read_text().splitlines() if path.exists() else []


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


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB. On CPU this is the number that matters."""
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 2**20 if platform.system() == "Darwin" else kb / 2**10
