"""The step on a real GPU, at the post's layer shape: H=4096, I=11008, V=32000, S=4096, B=1.

Needs CUDA. Four parts:
  1. each op of one block, timed, next to the accounting model's FLOPs and bytes
  2. a full step of a BF16 TinyLLM with fewer layers (default 8, 1.88B parameters): the allocator's
     ledger, the memory curve, forward / backward / optimizer times and MFU, without and with
     full activation checkpointing
  3. the memory wall: the 6.7B model's 16 bytes per parameter on one card
  4. host-to-device copies, pageable vs pinned, alone and next to a GEMM

    python labs/inside-a-training-step/scripts/08_gpu_step.py --layers 8
"""

import _path  # noqa: F401
import argparse

import torch

from stepanatomy import accounting as A
from stepanatomy import gpustep as G
from stepanatomy.report import Report

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--no-compile", action="store_true")
a = ap.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("08_gpu_step.py needs a CUDA GPU")

r = Report("08_gpu_step", "cuda")
s1 = A.LLAMA2_7B

r("## 1. One block, op by op, BF16, B=1, S=4096, H=4096, I=11008. Median of 20 runs")
rows = G.op_table(s1, compile_elementwise=not a.no_compile)
r.table(["op", "ms", "TFLOP/s", "GB/s", "model FLOP/byte", "bound (model)"],
        [[x.get(k, "") for k in ("op", "ms", "TFLOP/s", "GB/s", "model FLOP/byte", "bound (model)")] for x in rows])
eager = [x for x in rows if "compile" not in x["op"]]
tot = sum(x["ms"] for x in eager)
proj = sum(x["ms"] for x in eager if x["op"] in A.PROJECTIONS)
cheap = sum(x["ms"] for x in eager if x["op"] not in A.PROJECTIONS and "attention" not in x["op"])
ops = A.block_ops(s1)
cheap_flops = 100 * sum(o.flops for o in ops if o.name not in A.PROJECTIONS and "attention" not in o.name) / sum(
    o.flops for o in ops)
r(f"forward time of the block: {tot:.2f} ms. Projections {proj / tot:.0%}, attention"
  f" {(tot - proj - cheap) / tot:.0%}, norms + adds + silu*mul {cheap / tot:.0%} of the time"
  f" (they are {cheap_flops:.3f}% of the FLOPs)")
r.data["ops"] = rows

s = G.post_layer_shape(a.layers)
r()
r(f"## 2. Full step, BF16 TinyLLM with L={a.layers} at the post's layer shape, MasterWeightAdam, median of {a.steps}")
for mode in ("none", "full"):
    out = G.step_report(s, mode, steps=a.steps)
    r(f"### activation checkpointing: {mode}")
    r(f"parameters {out['params']:,}; bytes per parameter: weights {out['bytes/param weights (bf16)']:.2f},"
      f" master + m + v {out['bytes/param master+m+v']:.2f}, grads {out['bytes/param grads']:.2f}")
    r(f"peak allocated {out['peak GiB']:.2f} GiB, reserved {out['reserved GiB']:.2f} GiB;"
      f" activations + grads + workspace at peak {out['activation+workspace peak GiB']:.2f} GiB")
    ms = out["ms"]
    r(f"forward {ms['forward']:.1f} ms, backward {ms['backward']:.1f} ms ({out['backward / forward']:.2f}x),"
      f" optimizer {ms['optimizer']:.1f} ms ({out['optimizer share']:.0%} of the step), step {out['step ms']:.1f} ms")
    r(f"MFU (6ND / step time / 989 TFLOP/s): {out['MFU (6ND)']:.1%}; counting attention too:"
      f" {out['MFU (6ND + attention)']:.1%}; forward+backward only: {out['MFU fwd+bwd only (6ND)']:.1%}")
    if mode == "none":
        r("memory curve (GiB allocated):")
        for label, gib in out["curve"]:
            r(f"  {gib:7.2f}  {label}")
    r.data[f"step_{mode}"] = out
r(f"post's prediction for the activations of {a.layers} layers: {A.activation_bytes(s) / 2**30:.2f} GiB (16H per token per layer)")
r()
r("## 3. The memory wall: the 6.7B model's states as flat buffers, one category at a time")
w = G.memory_wall()
for name, gib in w["held"]:
    r(f"  holding through {name:15s} {gib:6.1f} GiB")
r(f"failed at: {w['failed at']} (GPU has {w['gpu GiB']:.1f} GiB; the ledger needs"
  f" {A.model_state_bytes(A.LLAMA2_7B)['total'] / 2**30:.1f} GiB)")
r.data["wall"] = w
r()
r("## 4. Host to device, 256 MiB")
h = G.h2d()
r.table(["measurement", "value"], [[k, v] for k, v in h.items()])
r.data["h2d"] = h
r.save()
