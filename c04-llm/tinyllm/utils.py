"""Small helpers shared by scripts, benchmarks and tests."""

from __future__ import annotations

import random
import time

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """cuda > mps > cpu, unless a device is named explicitly."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    """What a benchmark ran on, for the first line of its output."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def synchronize(device: torch.device) -> None:
    """Wait for queued kernels; without this, GPU timings measure only the launch."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """with Timer(device) as t: ...; t.seconds"""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        synchronize(self.device)
        self.seconds = time.perf_counter() - self._start
