"""Timing helpers: CUDA-event timing, CUDA-graph capture, and a hardware line for every results file."""

from __future__ import annotations

import platform
import statistics
import sys
import time
from typing import Callable

import torch


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip().removesuffix(" Processor")
    except OSError:
        pass
    return platform.machine()


def hardware_line() -> str:
    """One line naming the device and software, with no host or user details."""
    if torch.cuda.is_available():
        dev = f"{torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}"
    else:
        dev = f"CPU only ({cpu_name()}), {torch.get_num_threads()} threads"
    return f"# Hardware: {dev}, torch {torch.__version__}, Python {sys.version.split()[0]}"


def command_line() -> str:
    """The command that produced a results file, with the path made relative to the repo."""
    argv = list(sys.argv)
    script = argv[0]
    if "labs/" in script:
        script = "labs/" + script.split("labs/", 1)[1]
    return "# Command: python " + " ".join([script] + argv[1:])


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ms(fn: Callable[[], object], warmup: int = 3, iters: int = 20) -> float:
    """Median wall time of fn() in milliseconds. On CUDA each call is bracketed by events."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        if torch.cuda.is_available():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def cuda_graph(fn: Callable[[], torch.Tensor]) -> Callable[[], None]:
    """Capture fn once and return a replay function. fn must read only from tensors it closes over.

    A replay launches the whole captured step as one unit, so it measures the GPU work without Python or
    per-kernel launch cost. Comparing eager against graph time puts a number on launch overhead.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def measured_bandwidth(gbytes: float = 4.0) -> float:
    """Achievable HBM bandwidth in bytes/s: a large device-to-device copy reads and writes every byte once."""
    n = int(gbytes * 1e9 / 2)
    src = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    ms = time_ms(lambda: dst.copy_(src), warmup=3, iters=10)
    del src, dst
    torch.cuda.empty_cache()
    return 2 * n * 2 / (ms / 1e3)


def measured_peak_flops(n: int = 8192) -> float:
    """Achievable dense bf16 matmul rate in FLOP/s, from one large square GEMM."""
    a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    ms = time_ms(lambda: a @ b, warmup=5, iters=20)
    return 2 * n**3 / (ms / 1e3)


class Report:
    """Print lines and keep them; save() writes them under a hardware line and the command."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.lines = [hardware_line(), command_line(), ""]

    def __call__(self, *parts: object) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self) -> None:
        if self.path:
            from pathlib import Path

            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_text("\n".join(self.lines) + "\n")
            print(f"wrote {self.path}")
