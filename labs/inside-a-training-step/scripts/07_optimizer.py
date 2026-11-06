"""The optimizer step is bandwidth: time AdamW and convert the time into bytes per second.

Useful traffic is 28 bytes per parameter (see stepanatomy.optbench). A for-loop implementation
launches several kernels per tensor, each a separate pass over memory, so it moves far more than
28 bytes and reaches a lower useful rate. 'master' is this lab's MasterWeightAdam on BF16 weights.

    python 07_optimizer.py                          # CPU, about 50M parameters
    python 07_optimizer.py --device cuda --params 1.07e9 --hidden 4096
"""

import _path  # noqa: F401
import argparse

from stepanatomy import accounting as A
from stepanatomy.optbench import FP32_ADAM_BYTES_PER_PARAM, time_adam
from stepanatomy.report import Report

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--params", type=float, default=5e7)
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--iters", type=int, default=5)
a = ap.parse_args()

r = Report("07_optimizer" + ("_gpu" if a.device.startswith("cuda") else "_cpu"), a.device)
r(f"## AdamW step, {a.params:.3g} parameters in tensors shaped like H={a.hidden} blocks, median of {a.iters}")
rows = []
for v in ("for-loop", "foreach", "fused", "master"):
    out = time_adam(v, a.params, a.hidden, a.device, a.iters)
    rows.append([v, out["params"], out["tensors"], out["ms"], out["useful GB/s"]])
    r.data[v] = out
r.table(["variant", "params", "tensors", "ms", "useful GB/s (28 B/param)"], rows)
r(f"arithmetic intensity: about {A.ADAM_FLOPS_PER_PARAM} FLOPs / {FP32_ADAM_BYTES_PER_PARAM} bytes ="
  f" {A.ADAM_FLOPS_PER_PARAM / FP32_ADAM_BYTES_PER_PARAM:.2f} FLOP/byte")
best = min(rows, key=lambda row: row[3])
r(f"at the fastest variant's rate, a 6.74B-parameter step would take"
  f" {6.738e9 * FP32_ADAM_BYTES_PER_PARAM / (best[4] * 1e9) * 1e3:.0f} ms (post: about 75 ms at 2.5 TB/s on an H100)")
r.save()
