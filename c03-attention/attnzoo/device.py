"""Device and thread helpers shared by scripts, tests and benchmarks."""

from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device: cuda > mps > cpu, unless `prefer` names one."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued kernels, so wall-clock timings measure the work and not the launch."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def be_polite(num_threads: int = 2) -> None:
    """Thirteen tutorials share one laptop; cap intra-op threads."""
    torch.set_num_threads(num_threads)
