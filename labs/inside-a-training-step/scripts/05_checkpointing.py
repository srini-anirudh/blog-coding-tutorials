"""Activation checkpointing: memory saved, FLOPs added, time added, gradients unchanged.

CPU, FP32. "Saved at the turn" counts tensors autograd holds for backward when the loss exists.
It does not see the short-lived tensors a checkpointed block rebuilds during backward: on top of the
saved total, full checkpointing briefly holds one block's activations again. 08_gpu_step.py measures
real peak memory, recompute included, with the CUDA allocator.
"""

import _path  # noqa: F401
import time

import torch

from stepanatomy import flops as F
from stepanatomy import lifetime, models, recompute
from stepanatomy.report import Report

r = Report("05_checkpointing")
s = models.SMALL
x = models.random_batch(s)


def grads(mode):
    m = recompute.apply(models.build(s), mode)
    m(x, labels=x).loss.backward()
    return [p.grad.clone() for p in m.parameters()]


def step_ms(mode, iters=7):
    m = recompute.apply(models.build(s), mode)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        m(x, labels=x).loss.backward()
        times.append(time.perf_counter() - t0)
        m.zero_grad(set_to_none=True)
    return sorted(times)[iters // 2] * 1e3


ref = grads("none")
rows = []
base = {}
for mode in ("none", "selective", "full"):
    tl = lifetime.track(recompute.apply(models.build(s), mode), x)
    fl = F.model_flops(recompute.apply(models.build(s), mode), x)["total"]
    ms = step_ms(mode)
    diff = max((a - b).abs().max().item() for a, b in zip(ref, grads(mode)))
    per_block = sum(v for k, v in tl.saved_by_tag.items() if k.startswith("block")) / s.layers
    if mode == "none":
        base = {"saved": tl.peak, "flops": fl, "ms": ms}
    rows.append([mode, tl.peak / 2**20, per_block / 2**20, base["saved"] / tl.peak, fl / base["flops"],
                 ms / base["ms"], diff])
    r.data[mode] = {"saved_at_turn_bytes": tl.peak, "per_block_bytes": per_block, "flops": fl, "fwd_bwd_ms": ms,
                    "max_grad_diff_vs_none": diff}
r(f"## TinyLLM (C4), L={s.layers}, H={s.hidden}, B={s.batch}, S={s.seq}, FP32, SDPA. Forward+backward, median of 7")
r.table(["mode", "saved at turn MiB", "per block MiB", "memory cut", "FLOPs x", "time x", "max |grad diff|"], rows)
r("full: each block keeps only its input. selective: the two RMSNorms and the SiLU-multiply are rerun,"
  " every GEMM output is kept, so FLOPs do not change. The LM head and loss are never checkpointed here,"
  " which caps the cut for this small, shallow model.")
r("Timings on a shared CPU are rough; the FLOP ratio is exact.")
r.save()
