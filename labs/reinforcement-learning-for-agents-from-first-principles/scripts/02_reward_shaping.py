"""Sections "Dense reward" and "The one shaping construction that is safe".

1. The optimal policy of each reward, found by finite-horizon value iteration.
2. The cycle test: shaping reward collected once around every cycle of the page graph.
3. Tabular Q-learning under each reward: does a learner actually find the loop?
4. Invariance: for random potentials Phi, potential shaping never changes the optimal first action.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlfirst.report import Report  # noqa: E402
from rlfirst.shaping import (PAGES, REWARDS, greedy_rollout, make_potential, q_learning, simple_cycles,  # noqa: E402
                             cycle_gain, value_iteration)

p = argparse.ArgumentParser()
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--horizon", type=int, default=30)
p.add_argument("--no-write", action="store_true")
args = p.parse_args()
g, H = args.gamma, args.horizon

rep = Report("02_reward_shaping", write=not args.no_write)
rep(f"gamma = {g}, step budget = {H}. Pages: {' -> '.join(PAGES)}")

rep("\n== 1. optimal policy of each reward (value iteration) ==")
rows = []
for name, r in REWARDS.items():
    path, ret, ok = greedy_rollout(r, g, H)
    shown = " ".join(path) if len(path) <= 8 else " ".join(path[:8]) + f" ... ({len(path) - 1} steps)"
    rows.append([name, ok, len(path) - 1, round(ret, 3), shown])
rep.table(["reward", "reaches checkout", "steps", "optimal return", "path"], rows)
rep.data["optimal"] = [dict(zip(["reward", "reaches_goal", "steps", "return"], r[:4])) for r in rows]

rep("\n== 2. cycle test: undiscounted shaping reward collected once around each cycle ==")
cycles = simple_cycles()
rows = [[" -> ".join(PAGES[i] for i in c)] + [round(cycle_gain(r, c), 4) for r in REWARDS.values()]
        for c in cycles]
rep.table(["cycle"] + list(REWARDS), rows)
rep("discounted (gamma = %.2f) gain around product -> cart -> product: " % g +
    ", ".join(f"{n} {cycle_gain(r, [3, 4, 3], g):+.4f}" for n, r in REWARDS.items()))
rep.data["cycles"] = {" -> ".join(PAGES[i] for i in c): {n: cycle_gain(r, c) for n, r in REWARDS.items()}
                      for c in cycles}

rep("\n== 3. tabular Q-learning, 2000 episodes, explore 0.2, 5 seeds: greedy policy afterwards ==")
rows = []
for name, r in REWARDS.items():
    res = [q_learning(r, g, H, 2000, seed=s) for s in range(5)]
    rows.append([name, np.mean([x[0] for x in res]), np.mean([x[1] for x in res])])
rep.table(["reward", "success rate", "mean steps"], rows)
rep.data["q_learning"] = {r[0]: {"success": r[1], "steps": r[2]} for r in rows}

rep("\n== 4. potential shaping with 200 random potentials Phi in [-5, 5]: optimal first actions unchanged? ==")
rng = np.random.default_rng(0)
_, q_sparse = value_iteration(REWARDS["sparse"], g, H)
best = [max(q, key=q.get) if q else None for q in q_sparse]
phis = [{pg: float(rng.uniform(-5, 5)) for pg in PAGES} for _ in range(200)]
for zero in (True, False):
    same = 0
    for phi in phis:
        _, q_pot = value_iteration(make_potential(phi, zero_at_timeout=zero), g, H)
        same += all((max(q, key=q.get) if q else None) == b for q, b in zip(q_pot, best))
    label = "Phi = 0 at the goal and at time-out" if zero else "Phi = 0 at the goal only"
    rep(f"{label}: {same} / 200 random potentials give the sparse reward's optimal action at every page")
    rep.data[f"random_potentials_same_policy_zero_at_timeout_{zero}"] = same
rep.save()
