"""Reversibility sets a confidence threshold, and oversight has a capacity.

    python labs/the-agent-evaluation-playbook/scripts/06_thresholds_and_oversight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenteval.oversight import TIERS, escalation_sweep, harm_formula, p_star, threshold_policies  # noqa: E402
from agenteval.report import Report  # noqa: E402

E_GRID = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)


def main() -> None:
    r = Report("06_thresholds_and_oversight")
    rng = np.random.default_rng(0)
    r.add("1. p* = L_eff / (U + L_eff), L_eff = min(c_r, L)")
    r.table(["tier", "U", "L", "c_r", "p*"],
            [[t, kw["U"], kw["L"], kw.get("c_r", "-") if "c_r" in kw else "-", p_star(**kw)] for t, kw in TIERS.items()],
            fmt="{:.4g}")
    r.add(f"   A 97%-confident agent facing an irreversible action with L/U = 100: p* = {p_star(1, 100):.4f}, "
          f"so it should {'act' if 0.97 > p_star(1, 100) else 'escalate'}.")
    r.add()

    r.add("2. Three acting policies on 100,000 calibrated actions per tier (confidence ~ Beta(19, 1), mean 0.95).")
    r.add("   Escalated actions are worth -0.01 (the reviewer's time). 'below-p* acts' is the share of the agent's")
    r.add("   own actions taken at a confidence below p*; 'escalation recall' is the share of below-p* actions it")
    r.add("   handed to a person.")
    rows = [[d["tier"], d["policy"], d["utility"], d["escalated"], d["below_threshold_acts"], d["escalation_recall"]]
            for d in threshold_policies(100_000, rng)]
    r.table(["tier", "policy", "utility/action", "escalated", "below-p* acts", "escalation recall"], rows)

    r.add("3. A reviewer with capacity: per-item catch rate q(e) = 0.9 / (1 + e / 0.2), cost 0.02 per escalation.")
    r.add("   200,000 actions, confidence ~ Beta(9, 1), so about 10% are wrong. Harm = errors that reach the world")
    r.add("   per action + 0.02 e. 'random' escalates a random subset (the post's model); 'ranked' escalates the")
    r.add("   least confident actions first.")
    rand = escalation_sweep(E_GRID, 200_000, rng, ranked=False)
    rank = escalation_sweep(E_GRID, 200_000, rng, ranked=True)
    eps = 1 - 0.9   # error rate under Beta(9, 1)
    rows = [[f"{a['e']:g}", a["q"], a["harm"], float(harm_formula(a["e"], eps)), a["caught_frac"], b["harm"],
             b["caught_frac"], b["precision"], b["recall"], b["f1"]] for a, b in zip(rand, rank)]
    r.table(["e", "q(e)", "harm random", "H(e) formula", "caught random", "harm ranked", "caught ranked",
             "precision", "recall", "F1"], rows)
    for name, sweep in (("random", rand), ("ranked", rank)):
        best = min(sweep, key=lambda d: d["harm"])
        r.add(f"   {name}: lowest harm {best['harm']:.4f} at e = {best['e']:g}; e = 0 gives {sweep[0]['harm']:.4f}; "
              f"e = 1 gives {sweep[-1]['harm']:.4f}")
    r.add("   Escalating everything is not the safe policy: past the optimum, each extra escalation dilutes the")
    r.add("   reviewer's attention more than it adds catches.")
    r.save()


if __name__ == "__main__":
    main()
