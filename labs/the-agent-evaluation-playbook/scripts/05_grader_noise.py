"""The grader is a noisy instrument: a floor, a ceiling, a compressed axis, and the runs that compression costs.

    python labs/the-agent-evaluation-playbook/scripts/05_grader_noise.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenteval.noise import grade, measured, runs_needed_normal, runs_needed_sim  # noqa: E402
from agenteval.report import Report  # noqa: E402

GRADERS = ((0.0, 0.0), (0.08, 0.15), (0.20, 0.25))


def main() -> None:
    r = Report("05_grader_noise")
    rng = np.random.default_rng(0)
    r.add("alpha = P(grader passes a wrong solution), beta = P(grader fails a correct one).")
    r.add()
    r.add("1. Measured score against true capability pi (100,000 graded runs per cell, then the formula)")
    rows = []
    for pi in (0.0, 0.3, 0.5, 0.75, 1.0):
        row = [f"pi={pi:g}"]
        for a, b in GRADERS:
            sim = grade(rng.random(100_000) < pi, a, b, rng).mean()
            row += [float(sim), float(measured(pi, a, b))]
        rows.append(row)
    hdr = [""] + [f"{lab} a={a:g},b={b:g}" for a, b in GRADERS for lab in ("sim", "formula")]
    r.table(hdr, rows)
    r.add("Floor at pi = 0 is alpha. Ceiling at pi = 1 is 1 - beta. Slope is 1 - alpha - beta.")
    r.add()

    r.add("2. Runs per agent to separate pi = 0.50 from pi = 0.60 (two-sided 5% test, 80% power)")
    r.add("   simulated: bisection over n, 20,000 simulated comparisons per n, runs graded by the noisy grader")
    rows, base_sim, base_norm = [], None, None
    for a, b in GRADERS:
        n_sim = runs_needed_sim(0.5, 0.6, a, b, rng)
        n_norm = runs_needed_normal(0.5, 0.6, a, b)
        base_sim, base_norm = base_sim or n_sim, base_norm or n_norm
        rows.append([f"a={a:g}, b={b:g}", 1 - a - b, float(n_sim), n_norm, n_sim / base_sim, n_norm / base_norm,
                     (1 - a - b) ** -2])
    r.table(["grader", "slope", "n simulated", "n normal approx", "ratio sim", "ratio normal",
             "(1-a-b)^-2"], rows, fmt="{:.2f}")
    r.add("The ratios track (1 - alpha - beta)^-2. They differ a little because noise also moves the measured")
    r.add("rates toward 0.5, where a proportion's variance is largest.")
    r.add()

    r.add("3. The ceiling eats headroom as agents improve (alpha = 0.08, beta = 0.15)")
    rows = []
    for pi in (0.3, 0.5, 0.75, 0.9):
        m = float(measured(pi, 0.08, 0.15))
        top = 1 - 0.15
        rows.append([f"pi={pi:g}", m, top - m, 1 - pi, (top - m) / (1 - m)])
    r.table(["", "measured", "measured headroom to ceiling", "true headroom",
             "share of visible gap that is closable"], rows)
    r.save()


if __name__ == "__main__":
    main()
