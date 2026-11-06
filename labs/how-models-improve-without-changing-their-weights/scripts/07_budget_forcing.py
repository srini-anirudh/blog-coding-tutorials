"""Budget forcing, s1-style, on a real frozen model (post section 9, Figure 5, Nuance 08).

Greedy decoding on the arithmetic benchmark. A stop attempt is a token that starts 'Answer' or
end-of-sequence. Forced cap: after N thinking tokens append '\\nAnswer:'. Forced extension: replace
the first W stop attempts with ' Wait,' and let the model continue. We measure the three things the
s1 paper separates: control (does the knob move token count), scaling (does accuracy follow), and
flips (problems that change from right to wrong, or wrong to right, against the natural stop).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ttc.arith import make_problems, parse_answer
from ttc.lm import Forcing, budget_force, chat_ids, load, token_sets
from ttc.report import Report, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--problems", type=int, default=0, help="keep only N problems (0 = all 96)")
ap.add_argument("--caps", default="16,32,64,128")
ap.add_argument("--waits", default="1,2,4,8")
ap.add_argument("--max-think", type=int, default=1024)
ap.add_argument("--device", default="auto")
ap.add_argument("--out", default="07_budget_forcing")
args = ap.parse_args()
dev = pick_device(args.device)
r = Report(args.out)
model, tok = load(args.model, dev)
ts = token_sets(tok)
probs = make_problems()
if args.problems:
    probs = probs[:: max(1, len(probs) // args.problems)][: args.problems]
seqs = [chat_ids(tok, p["prompt"]) for p in probs]
wait = tok.encode(" Wait,", add_special_tokens=False)
hmm = tok.encode(" Hmm,", add_special_tokens=False)

settings = [("natural stop", Forcing())]
settings += [(f"cap {c}", Forcing(cap=int(c))) for c in args.caps.split(",")]
settings += [(f"wait x{w}", Forcing(waits=int(w), wait_ids=wait)) for w in args.waits.split(",")]
settings += [("hmm x2", Forcing(waits=2, wait_ids=hmm))]
for _, f in settings:
    f.max_think = args.max_think

r(f"model {args.model}, greedy, {len(probs)} arithmetic problems, thinking limit {args.max_think} tokens")
r("setting        think tokens  accuracy  capped  right->wrong  wrong->right   (flips vs natural stop)")
base, summary, t0 = None, [], time.perf_counter()
for name, f in settings:
    rows = budget_force(model, seqs, f, ts)
    ans = [parse_answer("Answer:" + tok.decode(x["answer"], skip_special_tokens=True)) for x in rows]
    ok = [a == p["truth"] for a, p in zip(ans, probs)]
    think = sum(len(x["think"]) for x in rows) / len(rows)
    capped = sum(x["capped"] for x in rows)
    if base is None:
        base = ok
    neg = sum(b and not o for b, o in zip(base, ok))
    pos = sum(o and not b for b, o in zip(base, ok))
    r(f"{name:<14} {think:12.1f}  {sum(ok) / len(ok):8.3f}  {capped:6d}  {neg:12d}  {pos:12d}")
    summary.append({"setting": name, "think_tokens": think, "accuracy": sum(ok) / len(ok), "capped": capped,
                    "right_to_wrong": neg, "wrong_to_right": pos})
r(f"({time.perf_counter() - t0:.0f}s for all settings)")

r("\n== One question, three modes (the s1 counting example) ==")
q = [chat_ids(tok, "How many times does the letter r appear in the word raspberry? Think step by step. "
                    "Then write the final answer on its own line as 'Answer: <number>'.")]
traces = {}
for name, f in (("natural stop", Forcing()), ("wait x2", Forcing(waits=2, wait_ids=wait)), ("cap 16", Forcing(cap=16))):
    f.max_think = args.max_think
    row = budget_force(model, q, f, ts)[0]
    text = tok.decode(row["think"]) + " || Answer:" + tok.decode(row["answer"], skip_special_tokens=True)
    traces[name] = text
    r(f"-- {name} ({len(row['think'])} thinking tokens) --\n{text.strip()[-700:]}\n")
r("(truth: 3. '||' marks where the loop appended '\\nAnswer:'.)")
r.save({"model": args.model, "n_problems": len(probs), "settings": summary, "counting_traces": traces})
