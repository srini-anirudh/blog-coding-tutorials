"""Helpers for the benchmarks: isolated child processes, timing, peak memory."""

from __future__ import annotations

import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnzoo.device import synchronize  # noqa: E402


def timed(fn: Callable[[], object], device: torch.device, repeats: int = 3) -> float:
    """Median wall-clock milliseconds, with the device queue drained around each call."""
    times = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def peak_rss_bytes() -> int:
    """High-water mark of this process. macOS reports bytes, Linux reports kilobytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if platform.system() == "Darwin" else rss * 1024


class PeakMemory:
    """Extra memory a block needs at its peak.

    CPU: growth of the process high-water mark (each measurement runs in a fresh child).
    MPS: growth of the Metal driver allocation, which keeps freed blocks cached until
    empty_cache(), so it behaves like a high-water mark.
    CUDA: torch.cuda.max_memory_allocated.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __enter__(self) -> "PeakMemory":
        if self.device.type == "mps":
            torch.mps.empty_cache()
            self.base = torch.mps.driver_allocated_memory()
        elif self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            self.base = torch.cuda.memory_allocated()
        else:
            self.base = peak_rss_bytes()
        return self

    def __exit__(self, *exc) -> None:
        if self.device.type == "mps":
            now = torch.mps.driver_allocated_memory()
        elif self.device.type == "cuda":
            now = torch.cuda.max_memory_allocated()
        else:
            now = peak_rss_bytes()
        self.bytes = max(0, now - self.base)


def run_child(script: str, args: dict, timeout: float = 240.0) -> dict | None:
    """Run `script --child '<json>'` in a fresh process and return its JSON row, or None."""
    cmd = [sys.executable, script, "--child", json.dumps(args)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout:.0f} s"}
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"error": (out.stderr.strip().splitlines() or ["no output"])[-1][:100]}


def hardware_note(device: str, threads: int = 2) -> str:
    chip = "Apple M5"
    if device == "cpu":
        return f"{chip} CPU, {threads} threads, torch {torch.__version__}"
    if device == "mps":
        return f"{chip} GPU through MPS, torch {torch.__version__}"
    return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}"
