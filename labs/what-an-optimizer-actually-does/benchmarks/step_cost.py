"""Where a step costs you: AdamW is memory-bound, launches add up, Newton-Schulz is matmul-bound.

1. One big fp32 tensor, torch fused AdamW. Bytes moved per parameter: read g, p, m, v and write p, m, v,
   7 x 4 = 28 B (the post's 28 B uses bf16 grads and a bf16 weight copy, which also totals 28 B).
   Achieved bandwidth = 28 N / time.
2. Many tensors shaped like a transformer: optlab's Python loop, torch for-loop, foreach and fused.
3. Five Newton-Schulz steps on MLP-shaped matrices: time and achieved FLOP/s.

Run: python labs/what-an-optimizer-actually-does/benchmarks/step_cost.py           (CPU sizes)
     python labs/what-an-optimizer-actually-does/benchmarks/step_cost.py --gpu     (H100 sizes)
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optlab import Adam, newton_schulz
from optlab.common import Report


def timed(fn, dev, reps=20, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def params_like(shapes, dev):
    ps = []
    for s in shapes:
        p = torch.nn.Parameter(torch.randn(s, device=dev) * 0.02)
        p.grad = torch.randn_like(p) * 1e-3
        ps.append(p)
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if a.gpu else "cpu")
    rep = Report("step_cost_h100" if a.gpu else "step_cost_cpu")

    rep("## 1. Fused AdamW on one flat fp32 tensor")
    sizes = [2**24, 2**26, 2**28] if a.gpu else [2**20, 2**22, 2**24]
    out = {"flat": {}, "many": {}, "ns": {}}
    for n in sizes:
        ps = params_like([(n,)], dev)
        opt = torch.optim.AdamW(ps, lr=1e-3, fused=True)
        t = timed(opt.step, dev)
        out["flat"][n] = {"ms": t * 1e3, "GBps": 28 * n / t / 1e9}
        rep(f"N = {n / 1e6:7.1f} M  {t * 1e3:8.3f} ms  {28 * n / t / 1e9:8.1f} GB/s achieved")
        del ps, opt

    d, f, L = (1024, 2816, 24) if a.gpu else (256, 704, 8)
    shapes = [(32000 if a.gpu else 4096, d)]
    for _ in range(L):
        shapes += [(d, d)] * 4 + [(f, d), (f, d), (d, f), (d,), (d,)]
    n = sum(torch.Size(s).numel() for s in shapes)
    rep(f"\n## 2. {len(shapes)} tensors, {n / 1e6:.1f} M params (d={d}, ffn={f}, {L} layers): ms per step")
    impls = {
        "optlab.Adam (Python loop)": lambda ps: Adam(ps, lr=1e-3, weight_decay=0.1),
        "torch for-loop": lambda ps: torch.optim.AdamW(ps, lr=1e-3, foreach=False),
        "torch foreach": lambda ps: torch.optim.AdamW(ps, lr=1e-3, foreach=True),
        "torch fused": lambda ps: torch.optim.AdamW(ps, lr=1e-3, fused=True),
    }
    for name, make in impls.items():
        opt = make(params_like(shapes, dev))
        t = timed(opt.step, dev)
        out["many"][name] = t * 1e3
        rep(f"{name:<27} {t * 1e3:8.2f} ms   {28 * n / t / 1e9:7.1f} GB/s")
        del opt

    dtype = torch.bfloat16 if a.gpu else torch.float32
    rep(f"\n## 3. Five Newton-Schulz steps in {str(dtype).replace('torch.', '')}")
    mats = [(4096, 4096), (11008, 4096)] if a.gpu else [(1024, 1024), (2816, 1024)]
    for m, k in mats:
        g = torch.randn(m, k, device=dev)
        t = timed(lambda: newton_schulz(g, 5, dtype=dtype), dev, reps=10)
        s = min(m, k)
        flops = 5 * (4 * s * s * max(m, k) + 2 * s**3)
        out["ns"][f"{m}x{k}"] = {"ms": t * 1e3, "TFLOPs": flops / t / 1e12}
        rep(f"{m} x {k}: {t * 1e3:8.2f} ms  {flops / t / 1e12:7.2f} TFLOP/s achieved  "
            f"({flops / (m * k):,.0f} FLOP/param)")
    rep.save(out)


if __name__ == "__main__":
    main()
