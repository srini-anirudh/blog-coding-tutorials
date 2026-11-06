"""One causal attention call on every execution backend: latency, peak memory, scaling.

Run:  python c03-attention/benchmarks/bench_backends.py           (CPU)
      python c03-attention/benchmarks/bench_backends.py --flash   (CUDA GPU only, fails without one)
Inputs are fixed q, k, v with B = 1, H = 4, d = 32. Each cell runs in a fresh process.
A cell is skipped when the (N, N) score matrix would exceed the memory budget. The CUDA
FlashAttention and memory-efficient kernels do not exist on CPU, so only --flash measures them.
The CPU table's sdpa_flash row is PyTorch's CPU flash kernel, a different program.
"""

from __future__ import annotations

import argparse
import json
import sys

import _common
import torch

from attnzoo.backends import attend
from attnzoo.device import synchronize

H, D = 4, 32
BUDGET_BYTES = 8 * 2**30
PLANS = {
    "cpu": ["naive", "sdpa_math", "sdpa", "sdpa_flash", "tiled"],
    "mps": ["naive", "sdpa", "tiled"],
    "cuda": ["naive", "sdpa_math", "sdpa_efficient", "sdpa_flash", "flash", "tiled"],
}
MATERIALISES = {"naive", "sdpa_math"}  # these hold the whole (N, N) score matrix


def child(a: dict) -> None:
    device = torch.device(a["device"])
    n = a["n"]
    q, k, v = (torch.randn(1, H, n, D, device=device, dtype=_common.dtype_for(a["device"])) for _ in range(3))
    synchronize(device)
    with torch.inference_mode():
        with _common.PeakMemory(device) as mem:
            ms = _common.timed(lambda: attend(q, k, v, backend=a["backend"], causal=True), device, repeats=a["repeats"])
    print(json.dumps({"ms": ms, "peak": mem.bytes}))


def flash_route() -> str:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "flash = PyTorch FA2 through SDPA, because the flash-attn package is not installed"
    return f"flash = flash-attn {flash_attn.__version__}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--child")
    p.add_argument("--flash", action="store_true", help="measure the CUDA kernels; fails without a GPU")
    args = p.parse_args()
    if args.child:
        return child(json.loads(args.child))
    if args.flash and not torch.cuda.is_available():
        sys.exit("--flash needs a CUDA GPU, and none is visible. Nothing was measured.")

    devices = ["cuda"] if args.flash else ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    lengths = [512, 1024, 2048, 4096, 8192, 16384]
    for device in devices:
        dtype = "bf16" if device == "cuda" else "fp32"
        print(f"\n{_common.hardware_note(device)}: causal attention, H = {H}, d = {D}, {dtype}. ms per call / peak extra MB")
        if device == "cuda":
            print(flash_route())
        print(f"{'backend':16}" + "".join(f"{n:>17}" for n in lengths))
        for backend in PLANS[device]:
            cells = []
            for n in lengths:
                score_bytes = n * n * H * 4
                if (backend in MATERIALISES or device == "mps") and score_bytes > BUDGET_BYTES:
                    cells.append("over budget")
                    continue
                r = _common.run_child(__file__, {"device": device, "n": n, "backend": backend, "repeats": 3 if n <= 4096 else 1})
                cells.append(r["error"][:15] if "error" in r else f"{r['ms']:.1f} / {r['peak'] / 2**20:.0f}")
            print(f"{backend:16}" + "".join(f"{c:>17}" for c in cells), flush=True)
    if not args.flash:
        print("\nsdpa_efficient and flash: not yet measured. They need a CUDA GPU. On one, run this script with --flash.")


if __name__ == "__main__":
    main()
