"""Sections "Credit over fifty steps" and "How the field got finer": trajectory-level versus anchor-state credit.

Part 1 prints the four-rollout worked example (Figure 7 in miniature). Part 2 trains a tabular policy on a 7 x 7
grid with three advantage estimators and the same binary outcome reward, over five seeds.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlfirst.credit import CreditConfig, train, worked_example  # noqa: E402
from rlfirst.report import Report  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--iters", type=int, default=150)
p.add_argument("--no-write", action="store_true")
args = p.parse_args()
torch.set_num_threads(2)
rep = Report("05_step_credit", write=not args.no_write)

rep("== 1. worked example: corridor of 4 cells, goal at cell 3, gamma 0.9, one group of 4 rollouts ==")
ex = worked_example()
rep.table(list(ex[0]), [list(r.values()) for r in ex])
reload_a = next(r for r in ex if r["rollout"] == "A" and r["action"] == "reload")
rep(f"the reload inside winning rollout A: trajectory-level advantage {reload_a['A_traj']:+.3f}, "
    f"two-level advantage {reload_a['A_two_level']:+.3f}")
rep.data["worked_example"] = ex

cfg0 = CreditConfig(n=7, T=30, groups=4, lr=1.0, iters=args.iters)
opt_steps = 2 * (cfg0.n - 1)
rep(f"\n== 2. {cfg0.n} x {cfg0.n} grid, budget {cfg0.T} steps (shortest path {opt_steps}), {cfg0.groups} groups x "
    f"G = {cfg0.G}, SGD lr {cfg0.lr}, {args.iters} iterations, {args.seeds} seeds ==")
modes = {"trajectory": "trajectory-level, binary R",
         "trajectory_discounted": "trajectory-level, R = gamma^(length-1)",
         "anchor": "two-level: episode + anchor step (lam 1)"}


def first(rows, pred):
    return next((r["iter"] for r in rows if pred(r)), None)


t0 = time.time()
table, data = [], []
for mode, label in modes.items():
    per = []
    for seed in range(args.seeds):
        rows = train(CreditConfig(n=7, T=30, groups=4, lr=1.0, iters=args.iters, mode=mode, seed=seed))
        last = rows[-10:]
        per.append({"iter_success_0.9": first(rows, lambda r: r["success"] >= 0.9),
                    "iter_steps_le_opt+1": first(rows, lambda r: r["success"] >= 0.9 and
                                                 r["steps_when_won"] <= opt_steps + 1),
                    "final_success": np.mean([r["success"] for r in last]),
                    "final_steps_when_won": np.nanmean([r["steps_when_won"] for r in last]),
                    "final_wasted": np.mean([r["wasted_frac"] for r in last]),
                    "shared_frac": np.mean([r["shared_frac"] for r in rows])})

    def mean_iter(key):
        xs = [x[key] for x in per]
        return f"{np.mean(xs):.0f}" if all(x is not None for x in xs) else \
            f"{sum(x is None for x in xs)} of {len(xs)} never"
    agg = {"mode": label, "iter_success_0.9": mean_iter("iter_success_0.9"),
           "iter_steps_le_opt+1": mean_iter("iter_steps_le_opt+1")}
    for k in ["final_success", "final_steps_when_won", "final_wasted", "shared_frac"]:
        agg[k] = float(np.mean([x[k] for x in per]))
    agg["per_seed"] = per
    data.append(agg)
    table.append([label, agg["iter_success_0.9"], agg["iter_steps_le_opt+1"], agg["final_success"],
                  agg["final_steps_when_won"], agg["final_wasted"], agg["shared_frac"]])
rep.table(["advantage", "iters to success 0.9", f"iters to <= {opt_steps + 1} steps", "final success",
           "final steps (wins)", "final wasted-step frac", "steps in shared anchor"], table)
rep("'final' = mean of the last 10 iterations, then over seeds. Wasted step = reload or wall bump.")
rep(f"\nwall time {time.time() - t0:.1f} s")
rep.data["training"] = data
rep.save()
