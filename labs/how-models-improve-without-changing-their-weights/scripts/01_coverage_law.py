"""Why sampling has a scaling law: per-problem sigmoids add up to a power law (post section 4, Figure 2).

1. One problem: coverage 1 - (1-p)^k is a sigmoid in log k, centred near 0.69 / p.
2. A population with p ~ Beta(a, b): the uncovered fraction falls like k^(-a). We check the exact
   Beta formula, a 20,000-problem Monte Carlo, and the fitted log-log slope against a.
3. A hopeless fraction pi0 puts a floor under the failure curve.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ttc.coverage import (beta_uncovered, beta_uncovered_tail, coverage_one, half_coverage_k, loglog_slope,
                          sample_population, simulate_coverage)
from ttc.report import Report

r = Report("01_coverage_law")
rng = np.random.default_rng(0)
ks = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]

r("== One problem: a sigmoid, not a line ==")
r("p      k_half  " + "  ".join(f"k={k:<5d}" for k in ks))
for p in (0.1, 0.01, 0.001):
    r(f"{p:<6} {half_coverage_k(p):6.0f}  " + "  ".join(f"{c:7.3f}" for c in coverage_one(p, ks)))
r("Each row goes from near 0 to near 1 in about two decades of k, centred near 0.69/p.")

r("\n== A population p ~ Beta(a, b=3): failure fraction 1 - C(k) ==")
r("a     source        " + "  ".join(f"k={k:<5d}" for k in ks) + "   slope(k>=30)")
data = {}
big = [k for k in ks if k >= 30]
for a in (0.35, 0.70, 1.40):
    exact = [beta_uncovered(a, 3.0, k) for k in ks]
    tail = [beta_uncovered_tail(a, 3.0, k) for k in ks]
    p = sample_population(a, 3.0, 20000, 0.0, rng)
    mc = 1 - simulate_coverage(p, ks, rng)
    s_exact = loglog_slope(big, [beta_uncovered(a, 3.0, k) for k in big])
    s_mc = loglog_slope(big, [m for k, m in zip(ks, mc) if k >= 30])
    for name, row, s in (("exact Beta", exact, s_exact), ("k^-a tail", tail, -a), ("Monte Carlo", mc, s_mc)):
        r(f"{a:<5} {name:<12}  " + "  ".join(f"{v:7.4f}" for v in row) + f"   {s:+.3f}")
    data[str(a)] = {"exact": exact, "monte_carlo": list(map(float, mc)), "slope_exact": s_exact,
                    "slope_mc": s_mc}
r("The fitted slope approaches -a. No single problem obeys a power law; the mixture does.")

r("\n== A hopeless fraction pi0 = 12% (a = 0.70) ==")
p = sample_population(0.70, 3.0, 20000, 0.12, rng)
mc = 1 - simulate_coverage(p, ks, rng)
r("k:         " + "  ".join(f"{k:7d}" for k in ks))
r("1 - C(k):  " + "  ".join(f"{v:7.4f}" for v in mc))
r(f"floor = pi0 = 0.12. At k=10000 the failure fraction is {mc[-1]:.4f}; the straight line has bent flat.")
data["pi0_0.12"] = list(map(float, mc))
r.save({"ks": ks, "curves": data})
