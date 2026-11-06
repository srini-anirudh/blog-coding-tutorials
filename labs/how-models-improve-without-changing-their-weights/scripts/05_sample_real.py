"""Repeated sampling from a real frozen model on the arithmetic benchmark.

Draws k samples per problem at temperature 0.7 from Qwen2.5-0.5B-Instruct and stores, for every
sample, the parsed answer and its mean token log-prob (the model's own confidence). The weights
never change. Script 06 turns this file into coverage, selector and verifier curves; script 09
uses it for allocation. Run on a GPU for the full setting; --problems 16 --k 8 is a CPU smoke run.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ttc.arith import make_problems, parse_answer
from ttc.lm import chat_ids, load, sample
from ttc.report import RESULTS, Report, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--per-level", type=int, default=12)
ap.add_argument("--problems", type=int, default=0, help="keep only the first N problems (0 = all)")
ap.add_argument("--k", type=int, default=64)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--max-new", type=int, default=320)
ap.add_argument("--rows", type=int, default=512, help="sequences decoded together")
ap.add_argument("--device", default="auto")
ap.add_argument("--out", default="05_samples")
args = ap.parse_args()
dev = pick_device(args.device)
r = Report(args.out)
torch.manual_seed(0)

model, tok = load(args.model, dev)
problems = make_problems(args.per_level)
if args.problems:
    problems = problems[:: max(1, len(problems) // args.problems)][: args.problems]
eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None}
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
gen = torch.Generator(device=dev).manual_seed(0)

per_batch = max(1, args.rows // args.k)
t0, n_tokens, examples = time.perf_counter(), 0, []
for i in range(0, len(problems), per_batch):
    chunk = problems[i:i + per_batch]
    seqs = [chat_ids(tok, pr["prompt"]) for pr in chunk for _ in range(args.k)]
    toks, scores = sample(model, seqs, eos=eos, pad=pad, max_new=args.max_new,
                          temperature=args.temperature, generator=gen)
    for j, pr in enumerate(chunk):
        rows = range(j * args.k, (j + 1) * args.k)
        texts = [tok.decode(toks[q], skip_special_tokens=True) for q in rows]
        pr["answers"] = [parse_answer(t) for t in texts]
        pr["scores"] = [round(scores[q], 4) for q in rows]
        pr["lengths"] = [len(toks[q]) for q in rows]
        n_tokens += sum(pr["lengths"])
        if len(examples) < 4 and pr["level"] in (0, 3, 6, 7):
            examples.append({"expr": pr["expr"], "truth": pr["truth"], "text": texts[0]})
    done = min(i + per_batch, len(problems))
    print(f"{done}/{len(problems)} problems, {time.perf_counter() - t0:.0f}s", file=sys.stderr)

elapsed = time.perf_counter() - t0
r(f"model {args.model}, {len(problems)} problems, k = {args.k}, temperature {args.temperature}, "
  f"max {args.max_new} new tokens")
r(f"generated {n_tokens} tokens in {elapsed:.0f}s ({n_tokens / elapsed:.0f} tokens/s)")
r("level             mean p    problems with 0 correct")
for lv in sorted({p["level"] for p in problems}):
    ps = [sum(a == p["truth"] for a in p["answers"]) / args.k for p in problems if p["level"] == lv]
    name = next(p["level_name"] for p in problems if p["level"] == lv)
    r(f"{name:<18}{sum(ps) / len(ps):6.3f}    {sum(x == 0 for x in ps)}/{len(ps)}")
for ex in examples:
    r(f"\n-- sample for {ex['expr']} (truth {ex['truth']}) --\n{ex['text'][:600]}")
r.save()
keep = ("id", "level", "level_name", "expr", "truth", "answers", "scores", "lengths")
(RESULTS / f"{args.out}.json").write_text(json.dumps({
    "hardware": r.lines[0], "command": r.lines[1], "model": args.model, "k": args.k,
    "temperature": args.temperature, "problems": [{f: p[f] for f in keep} for p in problems]}) + "\n")
