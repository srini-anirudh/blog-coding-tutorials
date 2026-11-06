"""Section 1: the classical picture. Bias, variance and noise of polynomial fits, and a vacuous VC bound.

Run:
    python labs/the-puzzle-of-overparameterization/scripts/01_bias_variance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overparam.bits import vc_bound
from overparam.common import Report, table
from overparam.linear import bias_variance

out = Report("01_bias_variance")
cols = {"degree": "d", "bias2": ".4f", "variance": ".4f", "noise": ".4f", "total": ".4f"}
for redraws in (20, 2000):
    rows = bias_variance(list(range(1, 13)), redraws=redraws)
    best = min(rows, key=lambda r: r["total"])
    out(f"Figure 1 setting: n = 14 equispaced points, sigma = 0.24, g(x) = sin(2.3x) + 0.42x^2, {redraws} redraws")
    out(table(rows, cols))
    out(f"lowest expected test error: degree {best['degree']} ({best['total']:.4f})")
    out()
    out.data[f"redraws_{redraws}"] = rows

vc = vc_bound()
out("VC arithmetic, section 1: W = 11e6 weights, L = 20 layers, n = 50,000 images, VC = W L log2 W")
out(f"VC = {vc['vc_dim']:.3e}   VC / n = {vc['ratio']:.3e}   sqrt(VC / n) = {vc['bound_term']:.0f}")
out("A bound on a probability that exceeds 1 is vacuous.")
out.data["vc"] = vc
out.save()
