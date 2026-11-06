"""Horizon is per-step reliability: halving the hazard doubles H50, and so does repairing half the errors.

Episodes are simulated step by step: each step errs with probability h, a fraction rho of errors is caught and
repaired, and an H-step task succeeds if the first unrepaired error comes after step H. The last part breaks the
independence assumption and shows the fatter tail.

    python labs/the-agent-evaluation-playbook/scripts/04_horizon.py
"""

from __future__ import annotations

import sys
from math import log
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenteval.reliability import (empirical_h50, empirical_survival, first_failure_steps, h50_exact,  # noqa: E402
                                   survival_exact)
from agenteval.report import Report  # noqa: E402

N = 400_000


def main() -> None:
    r = Report("04_horizon")
    rng = np.random.default_rng(0)
    r.add(f"{N:,} simulated episodes per row.")
    r.add()
    r.add("1. H50 against the per-step hazard h and the repair fraction rho")
    rows = []
    for h, rho in ((0.04, 0.0), (0.02, 0.0), (0.01, 0.0), (0.005, 0.0), (0.02, 0.5), (0.02, 0.75)):
        steps = first_failure_steps(h, N, rng, rho=rho)
        rows.append([f"{h:g}", f"{rho:g}", empirical_h50(steps), h50_exact(h, rho), log(2) / (h * (1 - rho)),
                     float(empirical_survival(steps, 50)), float(survival_exact(h, 50, rho))])
    r.table(["h", "rho", "H50 simulated", "H50 exact", "ln2 / h_eff", "S(50) simulated", "S(50) exact"], rows,
            fmt="{:.2f}")
    r.add("Each halving of h doubles H50. rho = 0.5 at h = 0.02 gives the same H50 as h = 0.01 with no repair.")
    r.add()

    r.add("2. Correlated failures: 20% of episodes start with a bad early decision that triples the hazard")
    h = 0.02
    steps = first_failure_steps(h, N, rng, early_bad=0.2, mult=3.0)
    h50 = empirical_h50(steps)
    h_fit = 1 - 0.5 ** (1 / h50)            # the constant hazard that matches the observed H50
    rows = []
    for mult_of_h50 in (0.25, 0.5, 1, 2, 4, 8):
        H = int(round(mult_of_h50 * h50))
        rows.append([f"{mult_of_h50:g} x H50 = {H}", float(empirical_survival(steps, H)),
                     float(survival_exact(h_fit, H))])
    r.table(["task length H", "S(H) simulated", "S(H) constant-hazard fit at H50"], rows)
    r.add(f"H50 = {h50:.0f}. The constant-hazard model fitted at H50 underpredicts survival on long tasks and")
    r.add("overpredicts it on short ones: clustered failures fatten the tail, so a single hazard is the wrong summary.")
    r.save()


if __name__ == "__main__":
    main()
