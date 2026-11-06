"""Section 8: parameter space is not function space. Permute or rescale a trained ReLU net; the function
stays put, the sharpness does not.

Run:
    python labs/the-puzzle-of-overparameterization/scripts/07_symmetries.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overparam.common import Report, table
from overparam.symmetry import (log10_factorial, param_norm, permuted, regression_data, rescaled, small_mlp,
                                top_hessian_eigenvalue, train)

out = Report("07_symmetries")
x, y = regression_data()
grid = torch.linspace(-1, 1, 1000)[:, None]
model = small_mlp(64)
loss = train(model, x, y)
with torch.no_grad():
    ref = model(grid)
out(f"1 -> 64 -> 1 ReLU MLP trained on 64 noisy points, final training MSE {loss:.4f}")

perm = torch.randperm(64, generator=torch.Generator().manual_seed(1))
with torch.no_grad():
    dp = float((permuted(model, perm)(grid) - ref).abs().max())
out(f"permuted hidden units: max |f_perm - f| on 1,000 inputs = {dp:.2e}")
out(f"orderings of one width-4096 layer: 4096! = 10^{log10_factorial(4096):.0f}")
out()

rows = []
for c in (0.1, 0.3, 1.0, 3.0, 10.0):
    m = rescaled(model, c)
    with torch.no_grad():
        diff = float((m(grid) - ref).abs().max())
        mse = float(torch.nn.functional.mse_loss(m(x), y))
    rows.append({"c": c, "max_abs_function_change": diff, "train_mse": mse, "param_norm": param_norm(m),
                 "top_hessian_eig": top_hessian_eigenvalue(m, x, y)})
out("Layer 1 multiplied by c, layer 2 divided by c")
out(table(rows, {"c": "g", "max_abs_function_change": ".1e", "train_mse": ".6f", "param_norm": ".2f",
                 "top_hessian_eig": ".3g"}))
lo, hi = min(r["top_hessian_eig"] for r in rows), max(r["top_hessian_eig"] for r in rows)
out(f"same function, top Hessian eigenvalue ranges over {hi / lo:.0f}x")
out.data = {"train_mse": loss, "perm_max_diff": dp, "log10_4096_factorial": log10_factorial(4096), "rescale": rows}
out.save()
