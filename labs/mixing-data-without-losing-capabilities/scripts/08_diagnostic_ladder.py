"""Spurious forgetting and the four-rung diagnostic ladder (post sections 2 and 11).

Stage 1 trains 'add' with answers written right after '=' ("37+45=082"). Stage 2 trains only 'rev' and 'sort',
for 30 to 3,000 steps, in one of two styles:

  plain     "R3745=5473"     the same output convention as stage 1
  preamble  "R3745=>5473"    every stage-2 answer starts with '>', like a corpus whose responses share a preamble

Then 'add' is measured four ways, cheapest first:

  strict        free greedy generation must be exactly "082\n" (the original harness)
  rung 1        same generations, a lenient parser that strips leading '>' (change the harness, not the model)
  rung 2        fine-tune on 100 'add' examples for 40 steps, then strict again (a tiny re-alignment set)
  rung 3        rank the right answer against 9 near-miss answers by likelihood (is the knowledge still there?)
  rung 4        steps of 'add' training needed to get back within 2 points of the stage-1 strict score

    python labs/mixing-data-without-losing-capabilities/scripts/08_diagnostic_ladder.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixlab.model import make_model  # noqa: E402
from mixlab.runs import Report, n_procs, parallel_map  # noqa: E402
from mixlab.tasks import STOI, build_pool, decode, encode, prompt_and_answer  # noqa: E402
from mixlab.train import greedy_answers, token_losses, train  # noqa: E402

STAGE1 = 3000
STAGE2 = (30, 100, 300, 3000)
N_EVAL = 500
ADD = build_pool("add", 20000, seed=1)
ADD_EVAL = build_pool("add", N_EVAL, seed=2)
REALIGN = build_pool("add", 100, seed=3)
STAGE2_POOLS = {
    style: {t: build_pool(t, 20000, seed=1, style=">" if style == "preamble" else "") for t in ("rev", "sort")}
    for style in ("plain", "preamble")
}
BASE = {}


def generations(model):
    prompts, golds = [], []
    for k in range(N_EVAL):
        p, a = prompt_and_answer(ADD_EVAL, k)
        prompts.append(p)
        golds.append(decode(a))
    outs = [decode(o) for o in greedy_answers(model, prompts, max_new=6)]
    return outs, golds


def strict_and_lenient(model):
    outs, golds = generations(model)
    strict = sum(o == g for o, g in zip(outs, golds)) / N_EVAL
    lenient = sum(o.lstrip(">") == g for o, g in zip(outs, golds)) / N_EVAL
    preamble = sum(o.startswith(">") for o in outs) / N_EVAL
    return 100 * strict, 100 * lenient, 100 * preamble, outs[:3]


@torch.no_grad()
def ranking(model, n: int = 200):
    """Rank the true sum among 10 candidates by total log-likelihood of the answer in the stage-1 format."""
    model.eval()
    offsets = (0, 1, -1, 2, -2, 10, -10, 11, -11, 20)
    hits = 0
    for k in range(n):
        p, a = prompt_and_answer(ADD_EVAL, k)
        true = int(decode(a))
        cands = [(true + o) % 1000 for o in offsets]
        ids = torch.tensor([p + encode(f"{c:03d}\n") for c in cands])
        lab = torch.full_like(ids, -100)
        lab[:, len(p) - 1 : -1] = ids[:, len(p) :]
        nll = token_losses(model, ids, lab).sum(1)
        hits += int(nll.argmin() == 0)
    model.train()
    return 100 * hits / n


def relearn_steps(model, target, max_steps=1500, every=25):
    m = copy.deepcopy(model)
    done = 0
    while done < max_steps:
        train(m, {"add": ADD}, {"add": 1}, steps=every, seed=500 + done, warmup=1)
        done += every
        if strict_and_lenient(m)[0] >= target:
            return done
    return f">{max_steps}"


def run(cfg):
    style, steps = cfg
    model = copy.deepcopy(BASE["model"])
    train(model, STAGE2_POOLS[style], {"rev": 1, "sort": 1}, steps=steps, seed=7)
    strict, lenient, pre, sample = strict_and_lenient(model)
    rank = ranking(model)
    re = copy.deepcopy(model)
    train(re, {"add": REALIGN}, {"add": 1}, steps=40, batch=32, seed=9, warmup=1)
    realigned = strict_and_lenient(re)[0]
    relearn = relearn_steps(model, BASE["strict"] - 2)
    return {"strict": strict, "lenient": lenient, "preamble": pre, "sample": sample, "rank": rank,
            "realigned": realigned, "relearn": relearn}


def main() -> None:
    rep = Report("08_diagnostic_ladder", workers=n_procs())
    model = make_model(seed=0)
    train(model, {"add": ADD}, {"add": 1}, steps=STAGE1, seed=0)
    BASE["model"] = model
    s, _, _, _ = strict_and_lenient(model)
    BASE["strict"] = s
    rank0 = ranking(model)
    rep(f"stage 1: 'add' for {STAGE1} steps -> strict {s:.1f}%, ranking {rank0:.1f}%")
    rep(f"re-alignment set: {len(REALIGN)} examples, 40 steps of batch 32; relearning target: strict >= {s - 2:.1f}%")
    cfgs = [(style, n) for style in ("plain", "preamble") for n in STAGE2]
    out = dict(zip(cfgs, parallel_map(run, cfgs)))
    rep("")
    rep("'add' after stage 2 (rev + sort only), accuracy %; rung 4 = add-training steps to recover")
    rows = []
    for c in cfgs:
        r = out[c]
        rows.append([c[0], c[1], f"{r['preamble']:.0f}%", f"{r['strict']:.1f}", f"{r['lenient']:.1f}", f"{r['realigned']:.1f}",
                     f"{r['rank']:.1f}", r["relearn"]])
    rep.table(["stage-2 style", "stage-2 steps", "outputs start with '>'", "strict", "rung 1 lenient", "rung 2 realigned",
               "rung 3 ranking", "rung 4 steps"], rows)
    rep("")
    for c in cfgs:
        rep(f"sample generations, {c[0]} {c[1]}: {[o.replace(chr(10), '|') for o in out[c]['sample']]}")
    rep.data = {"stage1": {"strict": s, "rank": rank0}, **{f"{a}_{b}": out[(a, b)] for a, b in cfgs}}
    rep.save()


if __name__ == "__main__":
    main()
