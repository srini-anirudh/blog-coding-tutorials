"""Section 12: tiling manufactures reuse.

First a counted matmul in plain Python: the naive loop loads 2 n^3 elements from slow memory, a T x T tiled loop
loads 2 n^3 / T, and both give the same C. Then, on the GPU, a naive CUDA kernel against shared-memory tiled kernels
and cuBLAS (fp32, TF32 off). On the CPU, torch.mm gives the baseline.

    python labs/why-gpus-are-built-for-deep-learning/scripts/07_tiling.py --device cuda
    python labs/why-gpus-are-built-for-deep-learning/scripts/07_tiling.py --device cpu
"""

import random

import _path  # noqa: F401
import torch

from whygpu import kernels as K
from whygpu.hw import Report, bench, device_from_argv, set_cpu_threads
from whygpu.models import counted_matmul, matmul_traffic


def counted(rep: Report) -> None:
    n = 32
    rng = random.Random(0)
    a = [[rng.random() for _ in range(n)] for _ in range(n)]
    b = [[rng.random() for _ in range(n)] for _ in range(n)]
    ref, _ = counted_matmul(a, b, None)
    rows = []
    for tile in (None, 2, 4, 8, 16):
        c, loads = counted_matmul(a, b, tile)
        err = max(abs(c[i][j] - ref[i][j]) for i in range(n) for j in range(n))
        rows.append([tile or "naive", loads, matmul_traffic(n, tile), matmul_traffic(n, None) / loads, err])
    rep(f"Counted matmul, n = {n}, loads from slow memory:")
    rep.table(["tile", "loads counted", "2n^3/T", "reuse factor", "max |C - C_naive|"], rows)


def gpu(rep: Report) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 4096
    a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
    ref = a @ b
    c = torch.empty_like(ref)
    flops = 2 * n ** 3
    rows = []
    naive = K.get("mm_naive")
    cases = [("naive, one thread per C element", lambda: naive(grid=(n // 16, n // 16, 1), block=(16, 16, 1),
                                                              args=[a, b, c, n]), 1)]
    for t in (8, 16, 32):
        kt = K.get("mm_tiled", T=t)
        cases.append((f"shared-memory tile T = {t}",
                      lambda kt=kt, t=t: kt(grid=(n // t, n // t, 1), block=(t, t, 1), args=[a, b, c, n]), t))
    cases.append(("cuBLAS torch.mm, fp32, TF32 off", lambda: torch.mm(a, b, out=c), "-"))
    for label, fn, reuse in cases:
        sec = bench(fn, "cuda", reps=3, trials=3)
        err = ((c - ref).abs().max() / ref.abs().max()).item()
        rows.append([label, reuse, sec * 1e3, flops / sec / 1e12, f"{err:.1e}"])
    rep(f"\nn = {n}, fp32 on the GPU:")
    rep.table(["kernel", "reuse T", "ms", "TFLOP/s", "max rel err vs cuBLAS"], rows)
    rep("The naive kernel already gets some reuse from L1 and L2, so the measured gap is smaller than T.")


def cpu(rep: Report) -> None:
    set_cpu_threads()
    n = 2048
    a, b = torch.randn(n, n), torch.randn(n, n)
    sec = bench(lambda: torch.mm(a, b), "cpu", reps=2, trials=3)
    rep(f"\nn = {n}, fp32 torch.mm on the CPU: {sec * 1e3:.1f} ms, {2 * n ** 3 / sec / 1e9:.0f} GFLOP/s")


def main() -> None:
    device = device_from_argv()
    rep = Report(f"07_tiling_{'h100' if device == 'cuda' else 'cpu'}.txt", device)
    counted(rep)
    gpu(rep) if device == "cuda" else cpu(rep)
    rep.save()


if __name__ == "__main__":
    main()
