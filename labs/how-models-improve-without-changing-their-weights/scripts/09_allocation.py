"""Allocate, don't extend: water-filling a sample budget (post section 11, Figure 7, Nuance 09).

Part 1, synthetic: twelve problems with known p. Greedy water-filling, the closed form
b* ~ (1/p) ln(p / lambda), the peak at p = e * lambda, and the gain over a uniform split.
Part 2, real: per-problem success rates measured in script 05 are treated as the truth. We compare
uniform, oracle water-filling (knows p), water-filling with p predicted from the difficulty level on a
held-out half (a learned difficulty predictor, in miniature), and stop-when-solved (a perfect verifier
and no knowledge of p).
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ttc.allocate import closed_form, expected_solved, last_gain, stop_when_solved, uniform, water_fill
from ttc.report import RESULTS, Report

ap = argparse.ArgumentParser()
ap.add_argument("--samples", default=str(RESULTS / "05_samples.json"))
args = ap.parse_args()
r = Report("09_allocation")
rng = np.random.default_rng(0)
data = {}

ps = np.round(np.geomspace(0.9, 0.002, 12), 4)
r("== Part 1: twelve problems, p from 0.9 down to 0.002 ==")
r("p:        " + " ".join(f"{p:6.3f}" for p in ps))
for B in (60, 150, 400):
    alloc = water_fill(ps, B)
    lam = last_gain(ps, alloc)
    cf = [closed_form(p, lam, exact=True) for p in ps]
    peak_p = ps[int(np.argmax(alloc))]
    r(f"\nB = {B}: lambda = {lam:.4f}, e*lambda = {math.e * lam:.4f}, most-funded p = {peak_p}")
    r("greedy:   " + " ".join(f"{b:6d}" for b in alloc))
    r("formula:  " + " ".join(f"{b:6.1f}" for b in cf))
    eu, ew = expected_solved(ps, uniform(len(ps), B)), expected_solved(ps, alloc)
    r(f"expected solved: uniform {eu:.2f}, water-filling {ew:.2f}")
    data[f"synthetic_B{B}"] = {"alloc": alloc.tolist(), "lambda": lam, "closed_form": cf,
                               "uniform": eu, "water_fill": ew}
r("\nAllocation rises with difficulty, peaks just above the give-up threshold, then drops to zero.")

path = Path(args.samples)
if not path.exists():
    r(f"\n(no {path.name}: run script 05 for part 2)")
else:
    d = json.loads(path.read_text())
    K = d["k"]
    probs = d["problems"]
    p_true = np.array([sum(a == p["truth"] for a in p["answers"]) / K for p in probs])
    lvl = np.array([p["level"] for p in probs])
    train = np.arange(len(probs)) % 2 == 0
    test = ~train
    level_p = {lv: p_true[train & (lvl == lv)].mean() for lv in set(lvl.tolist())}
    p_pred = np.array([level_p[lv] for lv in lvl[test]])
    pt = p_true[test]
    r(f"\n== Part 2: {test.sum()} held-out real problems ({d['model']}), p measured from {K} samples ==")
    r("budget/problem  uniform  oracle water-fill  level-predicted water-fill  stop-when-solved")
    rows = []
    for per in (1, 2, 4, 8, 16):
        B = per * len(pt)
        u = expected_solved(pt, uniform(len(pt), B))
        o = expected_solved(pt, water_fill(pt, B))
        pr = expected_solved(pt, water_fill(p_pred, B))
        s = stop_when_solved(pt, B, rng, trials=200)
        r(f"{per:<15d} {u:7.2f}  {o:17.2f}  {pr:26.2f}  {s:16.2f}")
        rows.append({"per_problem": per, "uniform": u, "oracle": o, "predicted": pr, "stop_when_solved": s})
    r(f"(expected number solved out of {len(pt)}; 'truth' is the measured rate, so this is a model, not a rerun)")
    data["real"] = rows
r.save(data)
