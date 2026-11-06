"""Sections "The utilisation bound" and "The bill for it": simulated, not measured on a cluster.

1. Synchronous utilisation E[T] / E[max T] for log-normal durations, by tail weight sigma and batch size N.
2. Asynchronous workers with a queue: usable trajectories per time unit and their age, by staleness window k.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlfirst.report import Report  # noqa: E402
from rlfirst.rollouts import async_run, sync_throughput, sync_utilisation  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--rounds", type=int, default=2000)
p.add_argument("--no-write", action="store_true")
args = p.parse_args()
rng = np.random.default_rng(0)
rep = Report("06_rollout_engine", write=not args.no_write)
rep("Durations: T = exp(sigma Z), median 1 time unit. Training step time is treated as 0.")

rep(f"\n== 1. synchronous utilisation E[T] / E[max T], {args.rounds} rounds per cell ==")
sigmas, Ns = [0.25, 0.5, 1.0, 1.5], [8, 64, 512]
rows, util = [], {}
for s in sigmas:
    u = [sync_utilisation(s, N, args.rounds, rng) for N in Ns]
    util[s] = dict(zip(Ns, u))
    rows.append([s] + u)
rep.table(["sigma"] + [f"N = {N}" for N in Ns], rows)
rep.data["sync_utilisation"] = {str(s): {str(N): v for N, v in d.items()} for s, d in util.items()}

N = B = 64
rep(f"\n== 2. {N} workers, trainer steps every B = {B} finished trajectories, {args.rounds} trainer steps ==")
rows, d2 = [], []
for s in [0.5, 1.0, 1.5]:
    sync = sync_throughput(s, N, args.rounds, rng)
    rows.append([s, "sync barrier", sync, 0.0, 0.0, 0, 0])
    d2.append({"sigma": s, "mode": "sync", "throughput": sync})
    for k in [0, 1, 4, None]:
        a = async_run(s, N, B, k, args.rounds, rng)
        rows.append([s, f"async, k = {'inf' if k is None else k}", a["throughput"], a["dropped_frac"],
                     a["age_mean"], a["age_p95"], a["age_max"]])
        d2.append({"sigma": s, "mode": "async", "k": k, **a})
rep.table(["sigma", "mode", "usable traj / time", "dropped as stale", "age mean", "age p95", "age max"], rows)
rep("Async with k = inf reaches N / E[T] = N exp(-sigma^2 / 2), the ceiling for N busy workers.")
rep.data["async"] = d2
rep.save()
