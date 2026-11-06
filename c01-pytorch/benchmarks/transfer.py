"""Host <-> device copy bandwidth: pageable vs pinned, and what non_blocking does.

Run: python c01-pytorch/benchmarks/transfer.py [--device mps|cuda] [--json out.json]
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scratchnet.device import pick_device, synchronize

torch.set_num_threads(2)
SIZES_MB = [0.0625, 1, 4, 16, 64, 256]


def time_copy(fn, device, reps: int) -> float:
    """Median seconds per call. synchronize() before and after: copies are queued, not finished."""
    fn()
    synchronize(device)  # warm-up: first use allocates and caches device memory
    times = []
    for _ in range(reps):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    if device.type == "cpu":
        print("no accelerator found; nothing to transfer to")
        return

    rows = []
    print(f"device={device}  torch={torch.__version__}")
    print(f"{'size_MB':>8} {'H2D_pageable':>13} {'H2D_pinned':>11} {'D2H':>8}   (GB/s)")
    for mb in SIZES_MB:
        n = int(mb * 2**20 // 4)  # float32 elements
        host = torch.randn(n)
        pinned = host.pin_memory()
        dev = host.to(device)
        reps = 50 if mb <= 4 else 10
        t_page = time_copy(lambda: host.to(device), device, reps)
        t_pin = time_copy(lambda: pinned.to(device, non_blocking=True), device, reps)
        t_d2h = time_copy(lambda: dev.to("cpu"), device, reps)
        gbs = [host.nbytes / t / 1e9 for t in (t_page, t_pin, t_d2h)]
        rows.append({"size_mb": mb, "h2d_pageable": gbs[0], "h2d_pinned": gbs[1], "d2h": gbs[2]})
        print(f"{mb:>8g} {gbs[0]:>13.2f} {gbs[1]:>11.2f} {gbs[2]:>8.2f}")
        del host, pinned, dev

    print("\nwhat non_blocking=True returns (64 MB, pinned source):")
    x = torch.randn(16 * 2**20).pin_memory()
    x.to(device)
    synchronize(device)
    t0 = time.perf_counter()
    y = x.to(device, non_blocking=True)
    t_issue = time.perf_counter() - t0
    synchronize(device)
    t_done = time.perf_counter() - t0
    print(f"  call returned after {t_issue * 1e3:.3f} ms, copy finished after {t_done * 1e3:.3f} ms")
    del y

    if args.json:
        args.json.write_text(json.dumps({"device": str(device), "torch": torch.__version__, "rows": rows}))


if __name__ == "__main__":
    main()
