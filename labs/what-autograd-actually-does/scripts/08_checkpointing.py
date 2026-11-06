"""Gradient checkpointing from scratch: peak saved memory and recompute cost against the number of segments.

36 residual MLP layers (the post's figure uses 36), width 256, batch 64, float32. For each segment count k the
script reports the peak bytes held for the backward pass (saved tensors plus segment boundaries), the number of
layer evaluations, the median wall time of 5 runs, and the largest gradient difference from plain backpropagation.
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint_sequential  # noqa: E402

from autogradlab.checkpoint import backprop_segments, grads_of, make_chain, plain_backprop  # noqa: E402
from autogradlab.report import Report  # noqa: E402
from autogradlab.saved import SavedTensorLedger  # noqa: E402

rep = Report("08_checkpointing")
n, width, batch = 36, 256, 64
layers = make_chain(n, width)
x = torch.randn(batch, width)
boundary = batch * width * 4


def timed(fn, *a):
    fn(*a)  # warm-up
    times = []
    for _ in range(5):
        t = time.perf_counter()
        r = fn(*a)
        times.append((time.perf_counter() - t) * 1e3)
    r["ms"] = sorted(times)[2]  # median of 5
    return r


base = timed(plain_backprop, layers, x)
ref = grads_of(layers).clone()
per_layer = base["peak_bytes"] / n
rep(f"Plain backprop: peak {base['peak_bytes'] / 2**20:.2f} MiB, {base['layer_evals']} layer evaluations, "
    f"{base['ms']:.0f} ms.")
rep(f"One layer saves {per_layer / boundary:.1f} activations' worth; a boundary is 1 activation ({boundary / 1024:.0f} KiB).")

rows = []
for k in [1, 2, 3, 4, 6, 9, 12, 18, 36]:
    r = timed(backprop_segments, layers, x, k)
    diff = (grads_of(layers) - ref).abs().max().item()
    rows.append([k, f"{r['peak_bytes'] / 2**20:.2f}", f"{r['peak_bytes'] / base['peak_bytes']:.2f}", r["layer_evals"],
                 f"{r['ms']:.0f}", f"{diff:.1e}"])
rep(f"\nSegments k over {n} layers:")
rep.table(["k", "peak MiB", "vs plain", "layer evals", "ms", "max grad diff"], rows)
best = min(rows, key=lambda r: float(r[1]))
c = per_layer / boundary
rep(f"\nMeasured best k = {best[0]}. sqrt(n) = {math.sqrt(n):.0f}. The sqrt(n) rule assumes a layer's saved tensors cost")
rep(f"the same as a boundary. Here a layer saves {c:.1f} times more, so memory ~ k + {c:.1f} n / k, minimised at")
rep(f"k = sqrt({c:.1f} n) = {math.sqrt(c * n):.1f}. Recomputing every segment costs one extra forward pass (2n evals) for")
rep("every k. The gradients are bitwise identical to plain backprop.")

# torch.utils.checkpoint does the same thing; check it lands on the same peak and gradient.
seq = torch.nn.Sequential(*layers)
for p in seq.parameters():
    p.grad = None
inp = x.clone()
with SavedTensorLedger(exclude=list(seq.parameters())) as led:
    out = checkpoint_sequential(seq, 6, inp, use_reentrant=False)
    out.pow(2).mean().backward()
diff = (grads_of(layers) - ref).abs().max().item()
rep(f"\ntorch.utils.checkpoint.checkpoint_sequential, 6 segments: peak {led.peak / 2**20:.2f} MiB, "
    f"max grad diff {diff:.1e}.")
rep("Wall times are medians of 5 on a shared machine (load near 60 on 192 cores). Read them as rough.")
rep.save()
