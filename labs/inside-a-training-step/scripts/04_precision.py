"""Four precisions, not one: what goes wrong when the accumulator or the master copy is BF16."""

import _path  # noqa: F401

from stepanatomy import precision as P
from stepanatomy.report import Report

r = Report("04_precision")

r("## A length-4096 dot product of BF16 values in [0, 1). Relative error against float64")
d = P.dot_errors()
r.table(["accumulator", "relative error"], [[k, v] for k, v in d.items() if k != "exact"])
r(f"Adding 1.0 to a BF16 sum 4096 times gives {P.ones_sum_stall():.0f}: past 256 the gap between BF16 values is 2.")
r()
r("## 1000 updates of -1e-4 x w to a weight of 1.0")
u = P.update_survival()
r.table(["copy", "final value"], [[k, v] for k, v in u.items()])
r("The BF16 weight never moves: 1e-4 is below half its spacing at 1.0 (2^-8 = 3.9e-3).")
r()
r("## Adam, lr 3e-4, fitting a 64 x 64 linear layer whose weights start at 1.0 and should end within 1% of it")
rows = []
curves = {}
for recipe in ("fp32", "master", "pure_bf16"):
    losses = P.regression_run(recipe)
    curves[recipe] = losses[::50]
    rows.append([recipe, losses[0], losses[99], losses[299], losses[-1]])
r.table(["recipe", "loss @1", "@100", "@300", "@600"], rows)
r("pure_bf16: BF16 weights and BF16 Adam state. Every update rounds away, so the loss never changes.")
r("master: BF16 compute copy, FP32 master. It learns, then floors where the BF16 copy cannot express the"
  " target (weights 1 +- 0.01 sit on a grid of 2^-7 = 0.0078). The FP32 run has no floor.")
r.data["loss_curves_every_50"] = curves
r()
r("## Casting 100,000 gradient-sized values (log-uniform in 1e-10 .. 1e-2)")
f = P.range_flush()
r.table(["format", "fraction flushed to zero"], [[k, v] for k, v in f.items() if "max" not in k])
r(f"largest finite value: FP16 {f['fp16 max']:.0f}, BF16 {f['bf16 max']:.3g}."
  " BF16 has FP32's exponent, so it needs no loss scaler.")
r.save()
