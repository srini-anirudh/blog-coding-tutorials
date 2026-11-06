"""Sections 3-4: the interpolation threshold and double descent, model-wise, sample-wise and epoch-wise.

The model is the one behind the post's Figures 4-6: x ~ N(0, diag(1/j)), y = <beta*, x> + 0.45 noise,
n = 40, and a least-squares fit that may use only the first p coordinates. Test error is the exact excess
risk. Each number is a median over 40 draws of the training set.

Run:
    python labs/the-puzzle-of-overparameterization/scripts/03_double_descent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overparam.common import Report, table
from overparam.linear import epoch_wise, model_wise, sample_wise

out = Report("03_double_descent")

ps = [2, 5, 10, 15, 20, 30, 36, 38, 40, 42, 44, 50, 60, 80, 120, 200, 300, 420]
rows = model_wise(ps)
out("Model-wise: n = 40, min-norm least squares on the first p coordinates")
out(table(rows, {"p": "d", "risk_median": ".4f", "norm_median": ".3f", "smallest_sv_median": ".4f"}))
peak = max(rows, key=lambda r: r["risk_median"])
left = min((r for r in rows if r["p"] < 40), key=lambda r: r["risk_median"])
out(f"classical sweet spot p = {left['p']} (risk {left['risk_median']:.4f}); peak at p = {peak['p']} "
    f"(risk {peak['risk_median']:.2f}); p = 420 reaches {rows[-1]['risk_median']:.4f}")
out()

ns = [8, 16, 24, 32, 40, 48, 54, 58, 60, 62, 66, 72, 80, 96, 108]
srows = sample_wise(ns)
out("Sample-wise: p = 60 fixed, n grows through the threshold n = p")
out(table(srows, {"n": "d", "risk_median": ".4f"}))
out()

steps = [1, 3, 10, 30, 100, 300, 1000, 3000, 10_000, 100_000, 1_000_000]
erows, end = epoch_wise(steps)
out("Epoch-wise: n = 40, p = 44 fixed, gradient descent from zero with step 1 / s_max^2")
out(table(erows, {"steps": "d", "risk_median": ".4f", "train_mse_median": ".2e"}))
best = min(erows, key=lambda r: r["risk_median"])
out(f"best test risk {best['risk_median']:.4f} at {best['steps']} steps; the min-norm end point is {end:.4f}")
out("This linear model shows the descent and the rise. It has no second descent in time.")
out.data = {"model_wise": rows, "sample_wise": srows, "epoch_wise": erows, "epoch_wise_min_norm_end": end}
out.save()
