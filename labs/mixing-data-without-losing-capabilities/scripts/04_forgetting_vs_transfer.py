"""Whack-a-mole: sequential tuning forgets, a mixture compromises, specialists top out higher (post sections 1-2).

Four tasks, 1,500 steps of data each. Three ways to spend the same 6,000 steps:

  specialists  one model per task, 1,500 steps each (four models)
  sequential   one model, copy -> add -> rev -> sort, 1,500 steps per stage, a fresh optimiser per stage
  joint        one model, a uniform mixture of all four for 6,000 steps (each task still gets ~1,500 steps of data)

The diagnostic from the post: a drop in the joint run is negative transfer (the data is still in every batch);
a drop in the sequential run is forgetting (the data left the batch). Two seeds each.

    python labs/mixing-data-without-losing-capabilities/scripts/04_forgetting_vs_transfer.py
    python labs/mixing-data-without-losing-capabilities/scripts/04_forgetting_vs_transfer.py --hidden 16   # less capacity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixlab.model import make_model  # noqa: E402
from mixlab.runs import Report, n_procs, parallel_map  # noqa: E402
from mixlab.train import evaluate, make_pools, scores, train  # noqa: E402

ORDER = ("copy", "add", "rev", "sort")
STAGE = 1500
SEEDS = (0, 1)
TR, EV = make_pools(ORDER)
HIDDEN = {"h": 48}


def run(cfg):
    kind, arg, seed = cfg
    model = make_model(hidden=HIDDEN["h"], seed=seed)
    if kind == "specialist":
        train(model, TR, {arg: 1}, steps=STAGE, seed=seed)
        return scores(evaluate(model, EV))
    if kind == "joint":
        h = train(model, TR, {t: 1 for t in ORDER}, steps=arg, seed=seed, eval_pools=EV, eval_every=STAGE)
        return [scores(e) for e in h.evals]
    # sequential: evaluate every 250 steps so the erosion of earlier skills is visible as it happens
    trace = []
    for k, task in enumerate(ORDER):
        h = train(model, TR, {task: 1}, steps=STAGE, seed=seed * 10 + k, eval_pools=EV, eval_every=250)
        trace += [{"stage": task, "step": s, **scores(e)} for s, e in zip(h.steps, h.evals)]
    return trace


def mean(rows, t):
    return round(sum(r[t] for r in rows) / len(rows), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=48)
    HIDDEN["h"] = ap.parse_args().hidden
    name = "04_forgetting_vs_transfer" + ("" if HIDDEN["h"] == 48 else f"_hidden{HIDDEN['h']}")
    rep = Report(name, workers=n_procs())
    rep(f"model width {HIDDEN['h']}, {sum(p.numel() for p in make_model(hidden=HIDDEN['h']).parameters()):,} parameters")
    cfgs = [("specialist", t, s) for t in ORDER for s in SEEDS]
    cfgs += [("joint", 4 * STAGE, s) for s in SEEDS] + [("sequential", None, s) for s in SEEDS]
    out = dict(zip([(c[0], c[1], c[2]) for c in cfgs], parallel_map(run, cfgs)))

    spec = {t: mean([out[("specialist", t, s)] for s in SEEDS], t) for t in ORDER}
    seq = [out[("sequential", None, s)] for s in SEEDS]
    joint = [out[("joint", 4 * STAGE, s)] for s in SEEDS]

    rep(f"exact-match accuracy (%), mean of seeds {SEEDS}; every task has 1,500 steps of data in every setting")
    rows = [["specialist (own task only)", *(spec[t] for t in ORDER), round(sum(spec.values()) / 4, 1)]]
    for k, task in enumerate(ORDER):
        ends = [[r for r in tr if r["stage"] == task][-1] for tr in seq]
        rows.append([f"sequential, after stage {k + 1} ({task})", *(mean(ends, t) for t in ORDER), round(sum(mean(ends, t) for t in ORDER) / 4, 1)])
    for k in range(4):
        snap = [j[k] for j in joint]
        rows.append([f"joint mixture, step {(k + 1) * STAGE}", *(mean(snap, t) for t in ORDER), round(sum(mean(snap, t) for t in ORDER) / 4, 1)])
    rep.table(["setting", *ORDER, "mean"], rows)
    rep("")
    final_seq = [[r for r in tr if r["stage"] == ORDER[-1]][-1] for tr in seq]
    final_joint = [j[-1] for j in joint]
    rep("gap to specialist at the end (points): negative = below the specialist")
    rep.table(
        ["task", "sequential (forgetting)", "joint (transfer)"],
        [[t, round(mean(final_seq, t) - spec[t], 1), round(mean(final_joint, t) - spec[t], 1)] for t in ORDER],
    )
    rep("")
    rep("erosion grows with time away: 'add' accuracy during the stages after it (seed 0)")
    tr0 = seq[0]
    rep.table(["stage", "step in stage", "add", "copy"], [[r["stage"], r["step"], r["add"], r["copy"]] for r in tr0 if r["stage"] in ("rev", "sort")])
    rep.data = {"specialist": spec, "sequential": seq, "joint": joint}
    rep.save()


if __name__ == "__main__":
    main()
