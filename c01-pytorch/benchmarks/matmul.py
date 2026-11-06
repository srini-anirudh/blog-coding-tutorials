"""Matmul from Python loops to BLAS to the GPU, and the timing mistake everyone makes once.

Run: python c01-pytorch/benchmarks/matmul.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scratchnet.device import pick_device, synchronize
from scratchnet.matmul import mm_flops, mm_loops, mm_rows

torch.set_num_threads(2)


def bench(fn, device, reps: int = 5) -> float:
    fn()
    synchronize(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    accel = pick_device()
    print(f"{'impl':<22} {'n':>5} {'ms':>10} {'GFLOP/s':>9}")
    cases = [("mm_loops (python)", 32, "cpu", mm_loops), ("mm_rows (broadcast)", 256, "cpu", mm_rows)]
    for n in (256, 1024, 2048):
        cases.append(("torch.mm cpu x2 thr", n, "cpu", torch.mm))
    if accel.type != "cpu":
        for n in (256, 1024, 2048, 4096):
            cases.append((f"torch.mm {accel.type}", n, accel.type, torch.mm))
    for name, n, dev, fn in cases:
        a, b = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
        t = bench(lambda: fn(a, b), dev, reps=1 if fn is mm_loops else 5)
        gflops = mm_flops(n, n, n) / t / 1e9
        print(f"{name:<22} {n:>5} {t * 1e3:>10.3f} {gflops:>9.{4 if gflops < 1 else 1}f}")

    if accel.type != "cpu":
        print(f"\ntiming a 4096 x 4096 matmul on {accel.type} with and without synchronize():")
        a, b = torch.randn(4096, 4096, device=accel), torch.randn(4096, 4096, device=accel)
        torch.mm(a, b)
        synchronize(accel)
        t0 = time.perf_counter()
        c = torch.mm(a, b)
        t_issue = time.perf_counter() - t0
        synchronize(accel)
        t_real = time.perf_counter() - t0
        print(f"  without sync: {t_issue * 1e3:.3f} ms   with sync: {t_real * 1e3:.3f} ms")
        del c

    print("\nbatched shapes: [B, M, K] @ [B, K, N] -> [B, M, N]")
    a, b = torch.randn(8, 128, 64), torch.randn(8, 64, 32)
    print("  torch.bmm   ", tuple(torch.bmm(a, b).shape))
    print("  torch.matmul", tuple(torch.matmul(a, b[0]).shape), "<- [K, N] broadcast over the batch")


if __name__ == "__main__":
    main()
