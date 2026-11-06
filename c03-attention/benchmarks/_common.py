"""Helpers for the benchmarks: isolated child processes, timing, peak memory."""

from __future__ import annotations

import json
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnzoo.device import be_polite, hardware_note, synchronize  # noqa: E402, F401  (hardware_note is re-exported)

be_polite()  # the parent and every child use the same thread count, so the label is true for both


def dtype_for(device: str) -> torch.dtype:
    """CUDA runs in bf16, the only kind of float the FlashAttention kernels accept besides fp16."""
    return torch.bfloat16 if device == "cuda" else torch.float32


def timed(fn: Callable[[], object], device: torch.device, repeats: int = 3) -> float:
    """Median wall-clock milliseconds, with the device queue drained around each call.

    One extra call runs first and is not counted: the first call pays for thread start-up
    and first-touch page faults, which would otherwise land only on the long runs (repeats = 1).
    """
    fn()
    times = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def proc_status_bytes(field: str) -> int:
    """Read one memory field (VmRSS, VmHWM) from /proc/self/status. Linux only."""
    return int(re.search(rf"{field}:\s+(\d+) kB", Path("/proc/self/status").read_text()).group(1)) * 1024


def reset_peak_rss() -> None:
    """Linux: restart the high-water mark. Importing torch peaks high enough to hide small blocks."""
    Path("/proc/self/clear_refs").write_text("5")


def peak_rss_bytes() -> int:
    """High-water mark of this process. macOS reports bytes through getrusage."""
    if platform.system() == "Darwin":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return proc_status_bytes("VmHWM")


class PeakMemory:
    """Extra memory a block needs at its peak.

    CPU: growth of the process high-water mark (each measurement runs in a fresh child).
    On Linux the mark is reset first, so this is peak minus current memory at entry.
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
        elif platform.system() == "Linux":
            reset_peak_rss()
            self.base = proc_status_bytes("VmRSS")
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
