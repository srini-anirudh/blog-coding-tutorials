"""Part II, continued: order, chaos and the critical line for a random tanh network, and dynamical isometry.

    python labs/why-transformers-do-not-explode/scripts/02_signal_propagation.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from stablelab.common import Report, seed_all  # noqa: E402
from stablelab.signal import (chi1, correlation_fixed_point, critical_sigma_w, critical_tanh_point,  # noqa: E402
                              empirical_correlations, jacobian_singular_values)


def fitted_depth_scale(cs: list[float], c_star: float) -> float:
    """Fit |c_l - c*| ~ exp(-l / xi) over the layers where the gap is still well above noise."""
    gap = np.abs(np.array(cs) - c_star)
    keep = np.where(gap > 0.05)[0]
    keep = keep[keep < 40]
    if len(keep) < 3:
        return float("nan")
    slope = np.polyfit(keep, np.log(gap[keep]), 1)[0]
    return -1 / slope


def main() -> None:
    seed_all(0)
    r = Report("02_signal_propagation")
    sb = 0.2
    sw_crit = critical_sigma_w(sb)
    r(f"1. Random tanh network, sigma_b = {sb}. Mean-field quantities by Gauss-Hermite quadrature (101 nodes).")
    r(f"   Critical sigma_w (chi_1 = 1) found by bisection: {sw_crit:.4f}")
    r("   Empirical depth scale: fit of |c_l - c*| ~ exp(-l/xi) for two inputs in width-1000 nets,")
    r("   mean of 4 seeds. c_0 = 0.2 when c* = 1, c_0 = 0.95 when c* < 1.")
    rows = []
    for sw in (0.8, 1.0, sw_crit, 1.5, 2.0, 3.0):
        x1 = chi1(sw, sb)
        c_star, slope = correlation_fixed_point(sw, sb)
        c0 = 0.2 if c_star > 0.99 else 0.95  # start far from c*, on the side the map moves away from
        cs = np.mean([empirical_correlations(sw, sb, depth=60, c0=c0, seed=s) for s in range(4)], axis=0)
        theory = -1 / math.log(slope) if 0 < slope < 1 else float("inf")
        phase = "ordered" if x1 < 0.999 else ("chaotic" if x1 > 1.001 else "critical")
        rows.append([round(sw, 4), phase, x1, c_star, theory, fitted_depth_scale(list(cs), c_star), cs[-1]])
    r.table(["sigma_w", "phase", "chi_1", "c*", "xi = -1/ln C'(c*)", "xi fitted", "c at layer 60"], rows)
    r("Ordered phase: c* = 1, every input maps to the same vector. Chaotic phase: c* < 1, nearby inputs decorrelate.")
    r("When c* = 1 the slope C'(c*) equals chi_1, so the post's 1/|ln chi_1| is the depth scale there. In the chaotic")
    r("phase the slope at c* < 1 is a different number, so 1/|ln chi_1| is only the ordered-side formula.")
    r("The fit matches theory within about 5% in the ordered phase. In the chaotic phase it comes out 1.6 to 1.8")
    r("times longer: the map is far from linear over the fitted range and width 1000 adds noise near c*.")
    r()

    r("2. Dynamical isometry: singular values s of the end-to-end Jacobian, width 200, one network each.")
    q_star = 0.05
    sw_t, sb_t = critical_tanh_point(q_star)
    r(f"   tanh on the critical line with q* = {q_star}: sigma_w = {sw_t:.4f}, sigma_b = {sb_t:.4f}.")
    r("   ReLU at its critical point sigma_w = sqrt(2), sigma_b = 0.")
    rows = []
    for act, sw, sbias in (("tanh", sw_t, sb_t), ("relu", math.sqrt(2), 0.0)):
        for weights in ("gaussian", "orthogonal"):
            for depth in (8, 32, 128):
                s = jacobian_singular_values(depth, 200, act, weights, sw, sbias)
                s2 = s**2
                near_one = ((s > 0.5) & (s < 2)).mean()
                rows.append([act, weights, depth, s2.mean(), s2.std() / s2.mean(), s.max(), near_one])
    r.table(["phi", "weights", "L", "mean s^2", "std/mean of s^2", "max s", "share of s in (0.5, 2)"], rows)
    r("Gaussian weights: the mean of s^2 stays of order 1 at criticality, but the spread grows with depth.")
    r("Orthogonal + tanh near the origin keeps the spectrum concentrated. Orthogonal + ReLU does not: the 0/1")
    r("derivative mask zeroes about half the directions at every layer (most s are below 1e-15), which is")
    r("Pennington et al.'s point that ReLU cannot be isometric. 'critical' row: xi is large but finite only because")
    r("the bisection lands a hair off the line; the fit over 40 layers cannot see a depth scale that long.")
    r.save()


if __name__ == "__main__":
    main()
