"""Section 8: how the data gets built. Rejection sampling and what an outcome filter lets through, on real GSM8K.

Qwen2.5-0.5B-Instruct samples k solutions per GSM8K training question (temperature 0.8). A solution is kept if its
final number equals the gold answer (the outcome filter V in the post). We then ask, as k grows:
  how many solutions are kept; how many of those are distinct reasoning paths; and how many of the kept ones contain a
  false arithmetic statement (an arithmetic statement 'a op b = c' that is wrong). The last number is a lower bound on invalid reasoning:
  it catches arithmetic slips only, not wrong logic.
"""

import argparse
import random

import _common
import numpy as np
import torch
from datasets import load_dataset
from sftlab import arith, lm, verifiers as V
from transformers import AutoTokenizer

INSTR = "\n\nSolve step by step, showing each calculation like 3 * 4 = 12. End with 'The answer is N.'"


@torch.no_grad()
def sample(model, tok, questions, k, temperature, max_new, qbatch):
    outs = []
    for b in range(0, len(questions), qbatch):
        qs = questions[b:b + qbatch]
        prompts = [tok.apply_chat_template([{"role": "user", "content": q + INSTR}], tokenize=False, add_generation_prompt=True) for q in qs]
        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(model.device)
        torch.manual_seed(1000 + b)
        g = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=0, max_new_tokens=max_new,
                           num_return_sequences=k, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i in range(len(qs)):
            outs.append(txt[i * k:(i + 1) * k])
        print(f"  sampled {b + len(qs)}/{len(questions)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--qbatch", type=int, default=16)
    a = ap.parse_args()
    out = _common.Report("08_data_construction")
    tok = AutoTokenizer.from_pretrained(lm.QWEN)
    model = lm.load_model(lm.QWEN, dtype=torch.bfloat16).eval()
    d = load_dataset("openai/gsm8k", "main", split="train")
    idx = random.Random(0).sample(range(len(d)), a.n)
    qs = [d[i]["question"] for i in idx]
    gold = [float(d[i]["answer"].split("####")[-1].strip().replace(",", "")) for i in idx]
    samples = sample(model, tok, qs, a.k, a.temperature, a.max_new, a.qbatch)
    ok = [[V.check_math(s, g) for s in ss] for ss, g in zip(samples, gold)]
    neq = [[arith.n_wrong(s) for s in ss] for ss in samples]
    out(f"Rejection sampling on {a.n} GSM8K training questions, Qwen2.5-0.5B-Instruct, T = {a.temperature}, up to {a.max_new} new tokens, k = {a.k} per question.")
    allsamples = [s for ss in samples for s in ss]
    has_eq = np.mean([e[0] > 0 for row in neq for e in row])
    out(f"Share of solutions containing at least one parseable 'a op b = c' statement: {has_eq:.3f}. Mean such statements per solution: {np.mean([e[0] for row in neq for e in row]):.2f}.")
    out("")
    out(f"{'k':>3} {'pass@k':>8} {'kept per question':>18} {'distinct paths kept':>20} {'duplicate share':>16} {'kept with a false equation':>28} {'questions with >=1 kept':>24}")
    res = {}
    for k in [1, 2, 4, 8, 16]:
        if k > a.k:
            continue
        passk = np.mean([any(row[:k]) for row in ok])
        kept = [(s, e) for ss, row, es in zip(samples, ok, neq) for s, o, e in zip(ss[:k], row[:k], es[:k]) if o]
        # distinct paths per question
        distinct = 0
        for ss, row in zip(samples, ok):
            keys = {arith.path_key(s) for s, o in zip(ss[:k], row[:k]) if o}
            distinct += len(keys)
        nk = len(kept)
        fp = np.mean([e[1] > 0 for s, e in kept]) if kept else float("nan")
        out(f"{k:>3} {passk:8.3f} {nk / a.n:18.2f} {distinct / a.n:20.2f} {1 - distinct / max(1, nk):16.3f} {fp:28.3f} {np.mean([any(row[:k]) for row in ok]):24.3f}")
        res[k] = dict(pass_at_k=float(passk), kept_per_q=nk / a.n, distinct_per_q=distinct / a.n, fp=float(fp))
    out("")
    k = a.k
    acc1 = np.mean([r[0] for r in ok])
    all_ok = np.mean([o for row in ok for o in row])
    wrong_share_all = np.mean([e[1] > 0 for row in neq for e in row])
    wrong_if_ok = np.mean([e[1] > 0 for row, es in zip(ok, neq) for o, e in zip(row, es) if o])
    wrong_if_bad = np.mean([e[1] > 0 for row, es in zip(ok, neq) for o, e in zip(row, es) if not o])
    out(f"Single-sample accuracy {all_ok:.3f}. Share with a false equation: all samples {wrong_share_all:.3f}; answer-correct samples {wrong_if_ok:.3f}; answer-wrong samples {wrong_if_bad:.3f}.")
    # step-level filter removes them
    kept_clean = [1 for ss, row, es in zip(samples, ok, neq) for s, o, e in zip(ss, row, es) if o and e[1] == 0]
    kept_all = [1 for row in ok for o in row if o]
    out(f"Adding a step check (drop solutions with a false equation) keeps {len(kept_clean)} of {len(kept_all)} answer-correct solutions ({len(kept_clean) / max(1, len(kept_all)):.1%}).")
    out("")
    ex = [(q, s, e) for q, ss, row, es in zip(qs, samples, ok, neq) for s, o, e in zip(ss, row, es) if o and e[1] > 0][:2]
    for q, s, e in ex:
        wrong = [x for x in arith.equations(s) if not arith.holds(x)]
        out(f"Example of an answer-correct solution with a false equation: {wrong[:2]}")
        out("  " + s.strip().replace("\n", " ")[:400])
    out("")
    out("Reading: the kept set grows with k, and so does the number of distinct paths, but more slowly (the rest repeat a path already")
    out("kept). The share of kept solutions with a false equation is small, 1% or less to a few percent, and it rests on a handful of")
    out("solutions per row (under 10 at k = 1 to 4). Sampling is not bit-reproducible on a GPU: across runs of this script (and its")
    out("parser while it was being written) the k = 1 value ranged from 0% to 5% and the k = 16 value from 0.5% to 1.6%. Do not read")
    out("a trend into it. The number is rough. The parser only sees plain 'a op b = c' statements; some hits are its own errors (a")
    out("sub-expression or algebra it cannot read) and some are real (an arithmetic slip that still ended at the right answer), while")
    out("wrong steps that are not arithmetic are invisible to it. It supports 'answer-correct solutions with a wrong step exist', not")
    out("a measured rate. One 0.5B model, 200 questions, one seed.")
    out.save()
    _common.save_json("08_data_construction", {"by_k": {str(k): v for k, v in res.items()}, "n_questions": a.n})


if __name__ == "__main__":
    main()
