"""Counterfactual oracle ablation on a parametric agent pipeline.

The pipeline has five seams. Each can be swapped for an oracle. The failure rates in BASE are chosen for the demo;
they are not measured from any agent. What the demo shows is the method: marginal deltas, their order dependence,
and an order-free summary (Shapley values) computed from the full lattice of 2^5 configurations.
"""

from __future__ import annotations

from itertools import combinations, permutations
from math import factorial

import numpy as np

COMPONENTS = ("retrieval", "tools", "memory", "planner", "verifier")
BASE = dict(p_search=0.92, p_tool=0.95, p_mem=0.95, p_plan=0.7, p_bad_route_dead=0.5, rho=0.3, h=0.03)


def success_rate(oracles: frozenset, n: int = 200_000, seed: int = 0, params: dict = BASE) -> float:
    """Simulate n episodes. Common random numbers (one seed for every configuration) keep the deltas low-noise.

    A good plan needs 1 search and 6 steps. A bad plan needs 3 searches and 10 steps, and half of bad routes cannot
    reach the goal at all. Every search, tool read, memory recall and execution step can fail. The verifier notices a
    failure and repairs it with probability rho; an oracle verifier has rho = 1. It cannot fix a dead route.
    """
    p = dict(params)
    if "retrieval" in oracles: p["p_search"] = 1.0
    if "tools" in oracles: p["p_tool"] = 1.0
    if "memory" in oracles: p["p_mem"] = 1.0
    if "planner" in oracles: p["p_plan"] = 1.0
    if "verifier" in oracles: p["rho"] = 1.0
    rng = np.random.default_rng(seed)
    u = rng.random((n, 33))
    plan_ok = u[:, 0] < p["p_plan"]
    reachable = plan_ok | (u[:, 1] >= p["p_bad_route_dead"])
    searches = np.where(plan_ok, 1, 3)
    steps = np.where(plan_ok, 6, 10)
    reads = steps // 2

    def any_unrepaired(count, p_fail, col):
        tries = u[:, col:col + 10]
        idx = np.arange(10)[None, :] < count[:, None]
        return ((tries < p_fail * (1 - p["rho"])) & idx).any(1)

    fail = (any_unrepaired(searches, 1 - p["p_search"], 2) | any_unrepaired(reads, 1 - p["p_tool"], 12) |
            (u[:, 22] < (1 - p["p_mem"]) * (1 - p["rho"])) | any_unrepaired(steps, p["h"], 23))
    return float((reachable & ~fail).mean())


def lattice(**kw) -> dict[frozenset, float]:
    return {frozenset(c): success_rate(frozenset(c), **kw)
            for r in range(len(COMPONENTS) + 1) for c in combinations(COMPONENTS, r)}


def marginal_deltas(lat: dict, order) -> list[tuple[str, float]]:
    """Add oracles one at a time in `order`; report each one's gain over the configuration before it."""
    have, out = frozenset(), []
    for c in order:
        out.append((c, lat[have | {c}] - lat[have]))
        have = have | {c}
    return out


def shapley(lat: dict) -> dict[str, float]:
    """Average marginal gain over all orders. The values sum to lat[all] - lat[none]."""
    n = len(COMPONENTS)
    return {c: sum(dict(marginal_deltas(lat, order))[c] for order in permutations(COMPONENTS)) / factorial(n)
            for c in COMPONENTS}
