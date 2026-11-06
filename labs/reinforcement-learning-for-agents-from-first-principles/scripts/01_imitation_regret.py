"""Section "Why cloning is not enough": measure the regret of cloning and of DAgger as the horizon T grows.

Prints, for each world (forgiving lane, cliff lane) and each eps, the regret of three learners at T = 10 ... 200,
the post's bound eps T (T - 1) / 2, and the log-log slope of regret against T. Slope 2 is the quadratic; slope 1 is
the linear interactive bound.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlfirst.imitation import bc_bound, fit_bc, fit_dagger, loglog_slope, regret  # noqa: E402
from rlfirst.report import Report  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=20000)
p.add_argument("--no-write", action="store_true")
args = p.parse_args()

Ts = [10, 20, 50, 100, 200]
rep = Report("01_imitation_regret", write=not args.no_write)
rep.data["runs"] = []
for world, cliff in [("cliff lane (|x| > 1 is absorbing)", 1), ("forgiving lane (no cliff)", None)]:
    for eps in [0.002, 0.01]:
        rep(f"\n== {world}, eps = {eps}, {args.episodes} evaluation episodes per cell ==")
        rows, curves = [], {"bc": [], "bc_recovery": [], "dagger": []}
        for T in Ts:
            rng = np.random.default_rng(T)
            width = T + 2
            tables = {"bc": fit_bc(T, width, rng, cliff=cliff),
                      "bc_recovery": fit_bc(T, width, rng, noise=0.1, cliff=cliff),
                      "dagger": fit_dagger(T, width, eps, rng, cliff=cliff)}
            row = [T]
            for name, table in tables.items():
                m, se = regret(table, T, eps, rng, n=args.episodes, cliff=cliff)
                curves[name].append(m)
                row += [f"{m:.4f} +- {se:.4f}", table.n_states]
            row.append(bc_bound(eps, T))
            rows.append(row)
        rep.table(["T", "BC regret", "BC states", "BC+recovery regret", "states", "DAgger regret", "states",
                   "eps T(T-1)/2"], rows)
        slopes = {k: loglog_slope(Ts, v) for k, v in curves.items()}
        rep("log-log slope of regret vs T:", ", ".join(f"{k} {v:.2f}" for k, v in slopes.items()))
        rep(f"ratio BC / DAgger at T = 50: {curves['bc'][2] / curves['dagger'][2]:.1f}")
        rep.data["runs"].append({"world": world, "eps": eps, "T": Ts, "regret": curves, "slope": slopes})
rep.save()
