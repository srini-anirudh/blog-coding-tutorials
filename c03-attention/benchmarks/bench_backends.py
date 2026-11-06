"""One causal attention call on every execution backend: latency, peak memory, scaling.

Run:  python c03-attention/benchmarks/bench_backends.py
Inputs are fixed q, k, v with B = 1, H = 4, d = 32. Each cell runs in a fresh process.
A cell is skipped when the (N, N) score matrix would exceed the memory budget. FlashAttention
kernels are CUDA-only: on CUDA add --flash; here that row is reported as not run.
"""

from __future__ import annotations

import argparse
import json

import _common
import torch

from attnzoo.backends import attend
from attnzoo.device import synchronize

H, D = 4, 32
BUDGET_BYTES = 2.2 * 2**30


def child(a: dict) -> None:
    torch.set_num_threads(2)
    device = torch.device(a["device"])
    n = a["n"]
    q, k, v = (torch.randn(1, H, n, D, device=device) for _ in range(3))
    synchronize(device)
    with torch.inference_mode():
        with _common.PeakMemory(device) as mem:
            ms = _common.timed(lambda: attend(q, k, v, backend=a["backend"], causal=True), device, repeats=a["repeats"])
    print(json.dumps({"ms": ms, "peak": mem.bytes}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--child")
    p.add_argument("--flash", action="store_true", help="also run the flash backend (CUDA only)")
    args = p.parse_args()
    if args.child:
        return child(json.loads(args.child))

    lengths = [512, 1024, 2048, 4096, 8192, 16384]
    plans = {
        "cpu": ["naive", "sdpa_math", "sdpa", "sdpa_flash", "tiled"],
        "mps": ["naive", "sdpa", "tiled"],
    }
    if args.flash and torch.cuda.is_available():
        plans = {"cuda": ["naive", "sdpa", "tiled", "flash"]}
    materialises = {"naive", "sdpa_math"}
    for device, backends in plans.items():
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        print(f"\n{_common.hardware_note(device)}: causal attention, H = {H}, d = {D}. ms per call / peak extra MB")
        print(f"{'backend':12}" + "".join(f"{n:>17}" for n in lengths))
        for backend in backends:
            cells = []
            for n in lengths:
                score_bytes = n * n * H * 4
                if (backend in materialises or device == "mps") and score_bytes > BUDGET_BYTES:
                    cells.append("over budget")
                    continue
                r = _common.run_child(__file__, {"device": device, "n": n, "backend": backend, "repeats": 3 if n <= 4096 else 1})
                cells.append(r["error"][:15] if "error" in r else f"{r['ms']:.1f} / {r['peak'] / 2**20:.0f}")
            print(f"{backend:12}" + "".join(f"{c:>17}" for c in cells), flush=True)
    if not args.flash:
        print("\nflash: not run. The flash-attn kernels need a CUDA GPU. On one, run this script with --flash.")


if __name__ == "__main__":
    main()
