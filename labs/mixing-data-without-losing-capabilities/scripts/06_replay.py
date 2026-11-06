"""Replay buys retention with adaptation (post section 12, figure 12).

Stage 1 trains 'add' to a high score. Stage 2 trains a new task for 1,500 steps with a fraction lambda of every
batch replayed from 'add'. Replay is compute-equivalent: each replayed example displaces a new one. Two new tasks:
'sub' shares the arithmetic machinery with 'add' (a weaker shift), 'rev' does not (a stronger shift).

    python labs/mixing-data-without-losing-capabilities/scripts/06_replay.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixlab.model import make_model  # noqa: E402
from mixlab.runs import Report, n_procs, parallel_map  # noqa: E402
from mixlab.train import evaluate, make_pools, scores, train  # noqa: E402

RATIOS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50)
NEW = ("sub", "rev")
STAGE1, STAGE2 = 3000, 1500
TR, EV = make_pools(("add", *NEW))
BASE = {}


def run(cfg):
    new, lam, seed = cfg
    model = copy.deepcopy(BASE["model"])
    train(model, TR, {new: 1 - lam, "add": lam}, steps=STAGE2, seed=100 + seed)
    s = scores(evaluate(model, {"add": EV["add"], new: EV[new]}))
    return s["add"], s[new]


def main() -> None:
    rep = Report("06_replay", workers=n_procs())
    model = make_model(seed=0)
    train(model, TR, {"add": 1}, steps=STAGE1, seed=0)
    BASE["model"] = model
    s0 = scores(evaluate(model, EV))
    rep(f"stage 1: 'add' only, {STAGE1} steps -> add {s0['add']}%, sub {s0['sub']}%, rev {s0['rev']}%")
    cfgs = [(n, lam, s) for n in NEW for lam in RATIOS for s in (0, 1)]
    out = dict(zip(cfgs, parallel_map(run, cfgs)))
    for new in NEW:
        rep("")
        rep(f"stage 2: new task '{new}', {STAGE2} steps, replay ratio lambda of 'add'; mean of 2 seeds, accuracy %")
        rows, best = [], None
        for lam in RATIOS:
            r = [out[(new, lam, s)] for s in (0, 1)]
            ret = sum(x[0] for x in r) / 2
            acq = sum(x[1] for x in r) / 2
            mm = min(ret, acq)
            if best is None or mm > best[1]:
                best = (lam, mm)
            rows.append([f"{100 * lam:.0f}%", f"{ret:.1f}", f"{acq:.1f}", f"{mm:.1f}", f"{s0['add'] - ret:+.1f}"])
            rep.data.setdefault(new, {})[str(lam)] = r
        rep.table(["replay", "retention (add)", f"acquisition ({new})", "min", "add lost (points)"], rows)
        rep(f"best max-min replay ratio for '{new}': {100 * best[0]:.0f}%")
    rep.data["stage1"] = s0
    rep.save()


if __name__ == "__main__":
    main()
