"""Section 5, Lab 1: the peak is a fragility. Tuned ridge removes it; so does removing label noise.

Run:
    python labs/the-puzzle-of-overparameterization/scripts/04_kill_the_peak.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overparam.common import Report, table
from overparam.linear import model_wise, tuned_ridge

out = Report("04_kill_the_peak")
ps = [5, 10, 20, 30, 36, 40, 44, 50, 60, 120, 420]
rows = {r["p"]: {"p": r["p"], "ridgeless": r["risk_median"]} for r in model_wise(ps)}
for lam in (1e-2, 1e-1):
    for r in model_wise(ps, ridge=lam):
        rows[r["p"]][f"ridge {lam:g}"] = r["risk_median"]
for r in tuned_ridge(ps):
    rows[r["p"]]["tuned ridge"] = r["risk_median"]
    rows[r["p"]]["best lambda"] = r["best_ridge"]
for r in model_wise(ps, sigma=0.0):
    rows[r["p"]]["sigma 0"] = r["risk_median"]
for r in model_wise(ps, sigma=0.1):
    rows[r["p"]]["sigma 0.1"] = r["risk_median"]
rows = list(rows.values())
out("Median excess risk over 40 draws, n = 40, sigma = 0.45 unless stated. 'tuned ridge' picks, per p, the best")
out("lambda on a 33-point log grid from 1e-6 to 1e2 by mean true risk (an oracle, as in Nakkiran et al. 2021).")
out(table(rows, {"p": "d", "ridgeless": ".3f", "ridge 0.01": ".3f", "ridge 0.1": ".3f", "tuned ridge": ".3f",
                 "best lambda": ".3g", "sigma 0.1": ".3f", "sigma 0": ".3f"}))
at40 = next(r for r in rows if r["p"] == 40)
out(f"at p = n = 40: ridgeless {at40['ridgeless']:.2f}, tuned ridge {at40['tuned ridge']:.3f}, "
    f"no noise {at40['sigma 0']:.2f}")
tuned = [r["tuned ridge"] for r in rows]
out(f"tuned-ridge curve: min {min(tuned):.3f}, max {max(tuned):.3f} (no peak; this data is not isotropic, "
    "so the monotonicity theorem does not strictly apply)")
out.data = {"rows": rows}
out.save()
