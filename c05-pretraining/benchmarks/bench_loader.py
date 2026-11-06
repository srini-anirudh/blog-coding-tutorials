"""Loader throughput with 0, 2 and 4 workers.

    python c05-pretraining/benchmarks/bench_loader.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretrain.loader import PackedWindows, ResumableLoader  # noqa: E402
from pretrain.shards import TokenShards  # noqa: E402
from pretrain.utils import resolve  # noqa: E402


def main() -> None:
    ds = PackedWindows(TokenShards(resolve("data_cache/tinystories"), "train"), 256)
    print(f"{len(ds):,} windows, batch 32, Apple M5 CPU")
    print(f"{'workers':>8}{'first batch ms':>16}{'batches/s':>11}{'tokens/s':>14}")
    for workers in (0, 2, 4):
        loader = ResumableLoader(ds, 32, seed=0, num_workers=workers, prefetch_factor=2)
        t = time.perf_counter()
        next(loader)
        first = time.perf_counter() - t
        n = 200
        t = time.perf_counter()
        for _ in range(n):
            next(loader)
        rate = n / (time.perf_counter() - t)
        print(f"{workers:>8}{1000 * first:>16.1f}{rate:>11.0f}{rate * 32 * 256:>14,.0f}")
        del loader


if __name__ == "__main__":
    main()
