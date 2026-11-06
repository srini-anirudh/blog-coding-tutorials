"""Figure 14: the five-condition ablation ladder on a real model, plus the "did it already know" control.

    python labs/memory-is-more-than-context/scripts/07_ladder.py
"""

from __future__ import annotations

import argparse
import random
import time

from _common import Report

from memlab.ladder import CONDITIONS, SYSTEM, decompose, make_world, memory_for, prompt_for
from memlab.lm import DEFAULT_MODEL, LM

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--model", default=DEFAULT_MODEL)
args = ap.parse_args()

out = Report("07_ladder", args.device)
lm = LM(args.model, device=args.device)
t0 = time.perf_counter()
results = []  # (seed, project, moved, condition, correct, tokens, gold_in_memory)
for seed in range(args.seeds):
    notes, questions = make_world(seed)
    rng = random.Random(seed)
    for q in questions:
        for cond in CONDITIONS:
            mem = memory_for(cond, q, notes, args.k, rng)
            c = lm.greedy_matches(prompt_for(q, mem), q.gold, system=SYSTEM)
            results.append((seed, q.project, q.moved, cond, c.correct, c.prompt_tokens,
                            any(q.gold in n.text for n in mem)))

n_q = len(results) // len(CONDITIONS)
out(f"{args.model}, {args.seeds} worlds x 24 projects = {n_q} questions per condition, k={args.k}, greedy")
out(f"{'condition':<12}{'accuracy':>10}{'moved':>8}{'unmoved':>9}{'gold in memory':>16}{'prompt tokens':>15}")
acc = {}
for cond in CONDITIONS:
    r = [x for x in results if x[3] == cond]
    acc[cond] = sum(x[4] for x in r) / len(r)
    moved = [x for x in r if x[2]]
    still = [x for x in r if not x[2]]
    out(f"{cond:<12}{acc[cond]:>10.3f}{sum(x[4] for x in moved) / len(moved):>8.3f}"
        f"{sum(x[4] for x in still) / len(still):>9.3f}{sum(x[6] for x in r) / len(r):>16.3f}"
        f"{sum(x[5] for x in r) / len(r):>15.0f}")
out("\nGaps (the post's decomposition):")
gaps = decompose(acc)
for name, v in gaps.items():
    out(f"  {name:<32}{v:+.3f}")

key = {(x[0], x[1]): x for x in results if x[3] == "no_memory"}
wins = [x for x in results if x[3] == "retrieved" and x[4]]
knew = sum(key[(x[0], x[1])][4] for x in wins)
out(f"\nControl: of {len(wins)} questions answered correctly with retrieved memory, {knew} are also answered")
out("correctly with the memory removed. Those were never memory wins.")
out(f"wall time {time.perf_counter() - t0:.0f} s")
out.save({"model": args.model, "k": args.k, "seeds": args.seeds, "accuracy": acc, "gaps": gaps,
          "retrieved_wins": len(wins), "already_knew": knew})
