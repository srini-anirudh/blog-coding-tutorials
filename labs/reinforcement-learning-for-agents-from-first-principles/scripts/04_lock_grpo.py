"""Sections "Now connect it to the horizon", "Which of the standard fixes actually address this" and "The horizon as
a curriculum variable": train group-relative RL on a combination lock and watch the informative fraction.

Part 1 sweeps the horizon with a binary reward. Part 2 fixes H = 40 and tries the post's interventions with the same
number of optimiser steps. Each configuration runs over three seeds.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlfirst.horizon import p_informative  # noqa: E402
from rlfirst.lock import LockConfig, train  # noqa: E402
from rlfirst.report import Report  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=300)
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--no-write", action="store_true")
args = p.parse_args()
torch.set_num_threads(2)
rep = Report("04_lock_grpo", write=not args.no_write)
base = LockConfig(iters=args.iters)
rep(f"K = {base.K} choices, {base.n_tasks} tasks, q0 = {base.q0}, SGD lr {base.lr}, {args.iters} iterations, "
    f"{args.seeds} seeds. 'success' is the exact expected binary success of the sampling policy at the target H.")


def run(**kw) -> dict:
    out = []
    for seed in range(args.seeds):
        cfg = LockConfig(iters=args.iters, seed=seed, **kw)
        log, _, _ = train(cfg)
        r = log.rows
        first_half = next((x["iter"] for x in r if x["target_sampled_success"] >= 0.5), None)
        out.append({"inf_first": r[0]["informative"], "inf_mean": np.mean([x["informative"] for x in r]),
                    "success": r[-1]["target_sampled_success"], "env_steps": r[-1]["env_steps"],
                    "backward_rollouts": sum(x["backward_rollouts"] for x in r), "iter_to_half": first_half})
    agg = {k: float(np.mean([o[k] for o in out])) for k in out[0] if k != "iter_to_half"}
    halves = [o["iter_to_half"] for o in out]
    agg["iter_to_half"] = "never" if any(h is None for h in halves) else int(np.mean(halves))
    agg["success_per_seed"] = [round(o["success"], 4) for o in out]
    return agg


t0 = time.time()
rep("\n== 1. binary reward, G = 8: horizon sweep ==")
rows, d1 = [], []
for H in [5, 10, 20, 40]:
    a = run(H=H)
    pred = p_informative(base.q0 ** H, base.G)
    rows.append([H, pred, a["inf_first"], a["inf_mean"], a["success"], a["iter_to_half"], a["env_steps"]])
    d1.append({"H": H, "predicted_inf_at_start": pred, **a})
rep.table(["H", "P_inf predicted at start", "measured (iter 0)", "mean over run", "final success",
           "iters to success 0.5", "env steps"], rows)
rep.data["horizon_sweep"] = d1

rep("\n== 2. H = 40: the interventions, same number of optimiser steps ==")
variants = {
    "binary, G = 8 (baseline)": {},
    "binary, G = 32": {"G": 32},
    "binary, G = 8, drop zero-variance groups": {"filter_zero": True},
    "graded reward, G = 8": {"graded": True},
    "horizon curriculum h0 5, +5 every 30 iters": {"curriculum": (5, 5, 30)},
}
rows, d2 = [], []
for name, kw in variants.items():
    a = run(H=40, **kw)
    rows.append([name, a["inf_mean"], a["success"], a["iter_to_half"], a["env_steps"], a["backward_rollouts"]])
    d2.append({"variant": name, **a})
rep.table(["variant", "mean informative fraction", "final success", "iters to 0.5", "env steps",
           "rollouts through backward"], rows)
rep.data["interventions"] = d2
rep(f"\nwall time {time.time() - t0:.1f} s")
rep.save()
