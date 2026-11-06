"""One label for every number a script prints, so a result always names its machine."""

from __future__ import annotations

import platform
from pathlib import Path

import torch


def use_threads(limit: int = 8) -> None:
    """Stay within `limit` CPU threads, and within whatever OMP_NUM_THREADS already allows."""
    torch.set_num_threads(min(limit, torch.get_num_threads()))


def hardware_label() -> str:
    """For example "AMD EPYC 7R13, 8 threads". The model name only, never a host name."""
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        names = [ln.split(":", 1)[1].strip() for ln in cpuinfo.read_text().splitlines() if ln.startswith("model name")]
        cpu = names[0].removesuffix(" Processor") if names else cpu
    return f"{cpu}, {torch.get_num_threads()} threads"
