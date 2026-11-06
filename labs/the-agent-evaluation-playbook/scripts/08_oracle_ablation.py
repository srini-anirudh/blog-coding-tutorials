"""Counterfactual ablation: swap one seam at a time for an oracle and re-run the same episodes.

The pipeline and its failure rates are a parametric toy chosen for the demo (see agenteval/ablation.py). The
method is the point: marginal deltas depend on the order, so report the order, or the full lattice.

    python labs/the-agent-evaluation-playbook/scripts/08_oracle_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenteval.ablation import BASE, COMPONENTS, lattice, marginal_deltas, shapley  # noqa: E402
from agenteval.report import Report  # noqa: E402


def main() -> None:
    r = Report("08_oracle_ablation")
    r.add(f"parameters (chosen, not measured): {BASE}")
    r.add("200,000 episodes per configuration, the same random numbers for every configuration.")
    r.add()
    lat = lattice()
    none, full = lat[frozenset()], lat[frozenset(COMPONENTS)]
    r.add(f"end-to-end success, no oracles: {none:.3f}; all five oracles: {full:.3f}")
    r.add()
    r.add("1. One oracle at a time, from the baseline")
    r.table(["oracle", "success", "delta"], [[c, lat[frozenset({c})], lat[frozenset({c})] - none] for c in COMPONENTS])

    r.add("2. Marginal deltas depend on the order the oracles are added")
    orders = {"information first": COMPONENTS, "planner and verifier first": ("planner", "verifier", "retrieval",
                                                                               "tools", "memory")}
    cols = {name: dict(marginal_deltas(lat, o)) for name, o in orders.items()}
    r.table(["oracle"] + list(orders) + ["Shapley (all 120 orders)"],
            [[c] + [cols[n][c] for n in orders] + [shapley(lat)[c]] for c in COMPONENTS])

    r.add("3. Non-additivity: the gain from oracle retrieval depends on what is already fixed")
    a = lat[frozenset({"retrieval"})] - none
    for other, why in (("planner", "a correct plan needs one search instead of three"),
                       ("verifier", "an oracle verifier already repairs every failed search")):
        b = lat[frozenset({"retrieval", other})] - lat[frozenset({other})]
        r.add(f"   retrieval oracle alone: +{a:.3f}; on top of a {other} oracle: +{b:.3f} ({why})")
    singles = sum(lat[frozenset({c})] - none for c in COMPONENTS)
    r.add(f"   sum of the five single-oracle deltas: {singles:.3f}; actual gain with all five: {full - none:.3f}")
    r.save()


if __name__ == "__main__":
    main()
