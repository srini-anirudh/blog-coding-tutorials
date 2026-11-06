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


def disable_cudnn_sdp() -> None:
    """Stop PyTorch from choosing the cuDNN attention backend for F.scaled_dot_product_attention.

    Measured on an H100 with PyTorch 2.15, bf16: with the default backend choice, generation is
    slow from the 48th token on, with and without the cache (tiny model: about 50 ms per step
    instead of about 5 ms; SmolLM2 size: about 73 ms instead of about 24 ms). A profile of one
    cached decode step showed aten::_cudnn_attention_forward using about 9.7 ms of CPU and
    4 microseconds of GPU time. The cause was not found, and the data does not show that the
    decode shape is the trigger. Call this once before generating on CUDA.
    See results/bench_generation_gpu_cudnn_default.txt.
    """
    torch.backends.cuda.enable_cudnn_sdp(False)


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
