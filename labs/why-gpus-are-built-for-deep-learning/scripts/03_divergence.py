"""Section 7: SIMT divergence, measured next to the post's pass-counting model (Figures 08 and 09).

Every thread runs the same amount of work: one of P path bodies, each a loop of dependent FMAs. Only the mapping
from thread to path changes. The model predicts the slowdown as the number of distinct paths per warp.
The second part does the same for a loop whose trip count varies per thread.

    python labs/why-gpus-are-built-for-deep-learning/scripts/03_divergence.py
"""

import random

import _path  # noqa: F401
import torch

from whygpu import kernels as K
from whygpu.hw import Report, bench
from whygpu.models import WARP, assign_paths, divergence_passes, tripcount_cost

ITERS = 2048
BLOCK = 256


def run_paths(threads: int, paths: list[int]) -> float:
    k = K.get("diverge")
    path = torch.tensor(paths, dtype=torch.int32, device="cuda")
    out = torch.empty(threads, device="cuda")
    return bench(lambda: k(grid=(threads // BLOCK, 1, 1), block=(BLOCK, 1, 1), args=[out, path, ITERS, 0.999]),
                 "cuda", reps=5, trials=5)


def main() -> None:
    rep = Report("03_divergence_h100.txt", "cuda")
    sms = torch.cuda.get_device_properties().multi_processor_count
    threads = sms * 2048 * 8  # eight full waves, so blocks are balanced across SMs as they finish
    warps = threads // WARP
    rep(f"{threads:,} threads ({sms} SMs x 2,048 x 8 waves), each runs one path body of {ITERS} dependent FMAs.\n")

    base = run_paths(threads, [0] * threads)
    rows = []
    for p in (2, 4, 8):
        for mode in ("warp", "tid", "random"):
            paths = assign_paths(threads, p, mode, seed=p)
            model = divergence_passes(paths) / warps
            t = run_paths(threads, paths)
            rows.append([p, mode, model, t / base, t * 1e3])
    rep(f"P = 1 (no divergence): {base * 1e3:.3f} ms")
    rep.table(["paths", "mapping", "model slowdown", "measured slowdown", "ms"], rows)
    rep("\nmapping: warp = path (tid / 32) % P, tid = path tid % P, random = path drawn per thread.")

    rep("\nA loop whose trip count varies per thread, uniform in [16, 4096] (same multiset of counts in both rows):")
    rng = random.Random(0)
    per_block = [16 * rng.randint(1, 2 * ITERS // 16) for _ in range(threads // BLOCK)]
    aligned = [n for n in per_block for _ in range(BLOCK)]  # same count for every thread of a block
    mixed = aligned[:]
    rng.shuffle(mixed)
    k = K.get("trips")
    out = torch.empty(threads, device="cuda")
    trows = []
    for name, iters in (("same count within each block", aligned), ("shuffled across warps", mixed)):
        n = torch.tensor(iters, dtype=torch.int32, device="cuda")
        t = bench(lambda: k(grid=(threads // BLOCK, 1, 1), block=(BLOCK, 1, 1), args=[out, n, 0.999]),
                  "cuda", reps=5, trials=5)
        trows.append([name, tripcount_cost(iters) / sum(iters) * WARP, t * 1e3])
    trows[1].append(trows[1][2] / trows[0][2])
    trows[0].append(1.0)
    rep.table(["mapping", "model cost / useful", "ms", "measured ratio"], trows)
    rep.save()


if __name__ == "__main__":
    main()
