"""Context, not parallelism: one long context versus a coordinator with workers, real model.

Lookup (breadth-first) and two-hop (interdependent) questions over a record store that grows
from about 450 to about 10,000 tokens. Every agent is the same small model. Every condition
reads every record exactly once per round, so prompt tokens are close to matched; the table
prints them.

    python labs/how-multi-agent-systems-actually-coordinate/scripts/07_context_split.py --device cuda
    python labs/how-multi-agent-systems-actually-coordinate/scripts/07_context_split.py --oracle   # perfect reader
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mascoord import haystack as H  # noqa: E402
from mascoord.lm import DEFAULT_MODEL, Agent  # noqa: E402
from mascoord.report import Report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--records", type=int, nargs="+", default=[32, 128, 384, 768])
ap.add_argument("--trials", type=int, default=40)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--slice", type=int, default=32, help="records per worker in the 'fixed slice' condition")
ap.add_argument("--oracle", action="store_true", help="use the rule-based perfect reader instead of a model")
args = ap.parse_args()

name = "07_context_split_oracle" if args.oracle else "07_context_split"
out = Report(name, "cpu" if args.oracle else args.device)
agent = None if args.oracle else Agent(args.model, device=args.device)
people = H.all_people()


def make_ask():
    """An ask() that also counts the prompt tokens this protocol spent."""
    bill = {"tokens": 0, "calls": 0}

    def ask(prompts):
        bill["calls"] += len(prompts)
        if agent is None:
            bill["tokens"] += sum(len(p.split()) for p in prompts)
            return H.oracle_ask(prompts)
        before = agent.prompt_tokens
        reps = agent.ask(prompts, max_new_tokens=12)
        bill["tokens"] += agent.prompt_tokens - before
        return reps
    return ask, bill


out(f"model: {'rule-based oracle' if args.oracle else args.model}, {args.trials} trials per cell, k = {args.k}")
out("token column: mean prompt tokens per question (whitespace words for the oracle)")
results = {}
for n in args.records:
    rng = random.Random(n)
    cells = {}
    t0 = time.time()
    k_slice = max(1, n // args.slice)
    for trial in range(args.trials):
        li = H.make_lookup(n, rng)
        th = H.make_two_hop(n, rng)
        runs = {
            ("lookup", "one agent"): lambda a: H.solo(li, a),
            ("lookup", f"{args.k} workers"): lambda a: H.split_one_round(li, a, args.k),
            ("lookup", f"{args.k} workers + evidence check"): lambda a: H.split_one_round(li, a, args.k, verify=True),
            ("lookup", f"{args.slice}-record slices"): lambda a: H.split_one_round(li, a, k_slice, verify=True),
            ("two-hop", "one agent"): lambda a: H.solo(th, a),
            ("two-hop", f"{args.k} workers, 1 round"): lambda a: H.split_one_round(th, a, args.k),
            ("two-hop", f"{args.k} workers, 2 rounds"): lambda a: H.split_two_round(th, a, args.k, people),
        }
        for key, fn in runs.items():
            ask, bill = make_ask()
            r = fn(ask)
            c = cells.setdefault(key, {"correct": [], "tokens": [], "calls": [], "conflict": [], "pos": []})
            c["correct"].append(bool(r["correct"]))
            c["tokens"].append(bill["tokens"])
            c["calls"].append(bill["calls"])
            c["conflict"].append(bool(r.get("conflict", False)))
            if key == ("lookup", "one agent"):
                c["pos"].append(li.meta["position"])
    secs = time.time() - t0
    out(f"\n{n} records ({secs:.0f} s for this block)")
    out(f"   {'task':<8} {'condition':<34} {'accuracy':>9} {'tokens':>8} {'calls':>6} {'conflicting msgs':>17}")
    for (task, cond), c in cells.items():
        acc = np.mean(c["correct"])
        se = np.sqrt(acc * (1 - acc) / len(c["correct"]))
        out(f"   {task:<8} {cond:<34} {acc:>5.2f}±{se:.2f} {np.mean(c['tokens']):>8.0f} "
            f"{np.mean(c['calls']):>6.1f} {np.mean(c['conflict']):>17.2f}")
    results[n] = {f"{t} / {c}": {"accuracy": float(np.mean(v["correct"])), "tokens": float(np.mean(v["tokens"])),
                                 "calls": float(np.mean(v["calls"])), "conflict": float(np.mean(v["conflict"]))}
                  for (t, c), v in cells.items()}
    pos = np.array(cells[("lookup", "one agent")]["pos"])
    ok = np.array(cells[("lookup", "one agent")]["correct"])
    terc = [ok[(pos >= a) & (pos < b)].mean() if ((pos >= a) & (pos < b)).any() else float("nan")
            for a, b in [(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)]]
    out(f"   one-agent lookup accuracy by answer position (start / middle / end): "
        + " / ".join(f"{x:.2f}" for x in terc))
    results[n]["solo_lookup_by_position"] = terc
    out.json({"model": "oracle" if args.oracle else args.model, "trials": args.trials, "k": args.k,
              "results": results})
if agent is not None:
    out(f"\nbill: {agent.bill()}")
