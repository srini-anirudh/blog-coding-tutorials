"""Sections "Where the signal goes", "Now connect it to the horizon" and "The one empirical regularity".

Every closed form is printed next to a Monte Carlo estimate.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlfirst.horizon import (group_size_needed, h50, h50_approx, h50_with_recovery, mc_informative,  # noqa: E402
                             mc_informative_graded, mc_informative_horizon, mc_success_with_recovery,
                             p_informative, p_success)
from rlfirst.report import Report  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--groups", type=int, default=50000, help="Monte Carlo groups per cell")
p.add_argument("--no-write", action="store_true")
args = p.parse_args()
rng = np.random.default_rng(0)
rep = Report("03_horizon_arithmetic", write=not args.no_write)
N = args.groups

rep("== 1. P_informative(p, G) = 1 - p^G - (1-p)^G against simulation, G = 8 ==")
rows = [[pp, p_informative(pp, 8), mc_informative(pp, 8, N, rng)] for pp in [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]]
rep.table(["p", "formula", "simulated"], rows)

q, G = 0.93, 8
rep(f"\n== 2. p = q^H and the informative fraction, q = {q}, G = {G} ==")
rows, data = [], []
for H in [5, 10, 20, 30, 50, 100]:
    ps = p_success(q, H)
    exact = p_informative(ps, G)
    sim = mc_informative_horizon(q, H, G, N, rng)
    graded = mc_informative_graded(q, H, G, N // 5, rng)
    need = group_size_needed(q, H, 0.5)
    rows.append([H, ps, exact, G * ps, sim, f"1 in {1 / exact:.1f}", graded, need])
    data.append({"H": H, "p": ps, "P_inf": exact, "G_p": G * ps, "simulated": sim, "graded_nonzero_var": graded,
                 "G_for_half": need})
rep.table(["H", "p = q^H", "P_inf exact", "G q^H", "P_inf simulated", "useful groups", "graded reward: var > 0",
           "G for P_inf >= 0.5"], rows)
rep("The post says a 100-step task gives 'one in forty' useful groups at G = 8, q = 0.93. The formula gives "
    f"1 in {1 / p_informative(p_success(q, 100), G):.0f}; one in forty would need q of about "
    f"{(1 / 40 / G) ** (1 / 100):.3f}.")
rep.data["informative"] = data

rep("\n== 3. the fifty-percent horizon H_50 = ln 0.5 / ln q, and what recovery does to it ==")
rows = []
for qq in [0.9, 0.93, 0.97, 0.99, 0.999]:
    rows.append([qq, h50(qq), h50_approx(qq)])
rep.table(["q", "H_50 exact", "0.69 / (1 - q)"], rows)
rep("\nq = 0.93, an error is caught (and the step retried) with probability r:")
rows, rec = [], []
for r in [0.0, 0.5, 0.8, 0.9]:
    h = h50_with_recovery(0.93, r)
    H_int = int(round(h))
    sim = mc_success_with_recovery(0.93, r, H_int, 20000, rng)
    rows.append([r, h, H_int, sim])
    rec.append({"r": r, "H50": h, "simulated_success_at_round_H50": sim})
rep.table(["catch rate r", "H_50 (closed form)", "H", "simulated success at H"], rows)
rep("Per-step error is 7% in every row. Only the catch rate changes, and H_50 grows by 10x.")
rep.data["recovery"] = rec
rep.save()
