"""GPU timing: warm up, then time a batch of back-to-back calls between two CUDA events, several times.

Back-to-back calls reuse whatever is still in L2. For bandwidth numbers that should mean HBM, use buffers much
larger than L2 (50 MB on H100); the bandwidth sweep uses small buffers on purpose to show the L2 shelf.
"""

import statistics

import torch


def cuda_time(fn, warmup: int = 5, iters: int = 20, reps: int = 5) -> float:
    """Median seconds per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / 1e3 / iters)
    return statistics.median(times)


def iters_for(seconds_per_call: float, budget: float = 0.05) -> int:
    """Enough calls to fill `budget` seconds, between 3 and 1000."""
    return int(min(1000, max(3, budget / max(seconds_per_call, 1e-7))))


def auto_time(fn, budget: float = 0.05) -> float:
    """Probe once, then pick the iteration count so each rep lasts about `budget` seconds."""
    probe = cuda_time(fn, warmup=2, iters=2, reps=1)
    return cuda_time(fn, warmup=3, iters=iters_for(probe, budget), reps=5)
