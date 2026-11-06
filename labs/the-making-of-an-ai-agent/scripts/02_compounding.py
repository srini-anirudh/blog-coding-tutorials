"""Section "The compounding-error argument": behaviour cloning vs DAgger in the corridor, cost against horizon.

    python labs/the-making-of-an-ai-agent/scripts/02_compounding.py
"""

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcraft.common import table, write_results
from agentcraft.compounding import exact_cost, fit_bc, fit_dagger, mean_cost, quadratic_approx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--episodes", type=int, default=20000)
    a = ap.parse_args()
    horizons = [3, 5, 10, 20, 50, 100]
    rows, data = [], []
    for H in horizons:
        rng = random.Random(H)
        bc = fit_bc(H, 50, rng)
        dagger = fit_dagger(H, a.eps, 50, 5, rng)
        dagger_irr = fit_dagger(H, a.eps, 50, 5, rng, irreversible=1.0)
        r = {"H": H, "exact_sum": exact_cost(a.eps, H), "eps_H2_over_2": quadratic_approx(a.eps, H),
             "bc": mean_cost(bc, H, a.eps, a.episodes, 1), "dagger": mean_cost(dagger, H, a.eps, a.episodes, 2),
             "dagger_irreversible": mean_cost(dagger_irr, H, a.eps, a.episodes, 3, irreversible=1.0),
             "eps_H": a.eps * H}
        data.append(r)
        rows.append([H, r["exact_sum"], r["eps_H2_over_2"], r["bc"], r["dagger"], r["dagger_irreversible"],
                     r["eps_H"], r["bc"] / H, r["dagger"] / H])

    def slope(key, h0=5, h1=50):
        c = {r["H"]: r[key] for r in data}
        return math.log(c[h1] / c[h0]) / math.log(h1 / h0)

    slopes = {k: slope(k) for k in ("bc", "dagger", "dagger_irreversible", "exact_sum")}
    text = (f"Per-step error eps = {a.eps}. {a.episodes} episodes per cell. BC: 50 expert demos. DAgger: 50 demos + "
            "5 rounds of 50 learner rollouts labelled by the expert.\nCost = expected number of steps that end off "
            "the expert's path.\n\n"
            + table(rows, ["H", "exact sum (absorbing)", "eps H(H+1)/2", "BC (measured)", "DAgger (measured)",
                           "DAgger, irreversible mistakes", "eps H", "BC cost / H", "DAgger cost / H"])
            + "\n\nLog-log slope of cost against H between H = 5 and H = 50 (2 = quadratic, 1 = linear):\n"
            + "\n".join(f"  {k}: {v:.2f}" for k, v in slopes.items()))
    print(text)
    write_results("02_compounding", text, {"eps": a.eps, "rows": data, "slopes_5_to_50": slopes})


if __name__ == "__main__":
    main()
