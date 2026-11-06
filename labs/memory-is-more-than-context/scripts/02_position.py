"""Figure 4: the same fact moved through the context, on a real model (key-value retrieval).

    python labs/memory-is-more-than-context/scripts/02_position.py                       # CPU, small grid
    python labs/memory-is-more-than-context/scripts/02_position.py --device cuda \\
        --pairs 10 30 75 --trials 40 --name 02_position_gpu                             # GPU grid

For each context size and position: the share of trials where greedy decoding returns the
right value, the mean log-probability of that value, and the attention share that the last
prompt token puts on the answering pair (mean over layers and heads). "restate" repeats the
answering pair just before the question, with the original pair left in the middle.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import torch

from _common import Report

from memlab.lm import DEFAULT_MODEL, LM
from memlab.position import kv_case, positions_for

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", type=int, nargs="+", default=[10, 30])
ap.add_argument("--trials", type=int, default=8)
ap.add_argument("--slots", type=int, default=5)
ap.add_argument("--device", default="cpu")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--name", default="02_position")
args = ap.parse_args()

out = Report(args.name, args.device)
lm = LM(args.model, device=args.device, eager=True)
t0 = time.perf_counter()
rows = []
out(f"{args.model}, key-value retrieval with random UUIDs, {args.trials} trials per cell, greedy")
out(f"{'pairs':>6}{'tokens':>8}{'position':>15}{'accuracy':>10}{'log p(value)':>14}{'attention on pair':>19}")
for n in args.pairs:
    cells = [(pos, False) for pos in positions_for(n, args.slots)] + [(positions_for(n, args.slots)[args.slots // 2], True)]
    for pos, restate in cells:
        rng = random.Random(1000 * n + pos)  # same random contexts for the plain and restated cell
        checks = []
        for _ in range(args.trials):
            case = kv_case(n, pos, rng, restate=restate)
            checks.append(lm.greedy_matches(case.prompt, [case.answer, f'"{case.answer}"'], needle=case.needle))
        row = {"pairs": n, "position": pos, "restate": restate,
               "tokens": round(statistics.mean(c.prompt_tokens for c in checks)),
               "accuracy": sum(c.correct for c in checks) / len(checks),
               "logprob": statistics.mean(c.logprob for c in checks),
               "attention": statistics.mean(c.needle_attention for c in checks)}
        rows.append(row)
        label = f"{pos + 1}/{n}" + (" +restate" if restate else "")
        out(f"{n:>6}{row['tokens']:>8}{label:>15}{row['accuracy']:>10.3f}{row['logprob']:>14.3f}{row['attention']:>19.4f}")
out(f"wall time {time.perf_counter() - t0:.0f} s")
out.save({"model": args.model, "trials": args.trials, "torch_threads": torch.get_num_threads(), "rows": rows})
