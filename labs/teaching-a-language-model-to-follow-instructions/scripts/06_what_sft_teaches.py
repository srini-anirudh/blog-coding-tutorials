"""Section 6: what does SFT actually teach?

A. Format alignment and elicitation. Format adherence against the number of SFT examples N, against few-shot
   prompting of the same base model, on held-out topics, and on a format the model was never trained on.
B. Knowledge acquisition (a small version of Gekhman et al.). Closed-book trivia questions split by whether the
   base model already answers them. Fine-tune on mixtures of known and unknown answers and watch how fast each is
   fitted and what happens to accuracy on questions the model knew.
"""

import argparse
import math
import random
import re
import string

import _common
import numpy as np
import torch
from sftlab import data, formats, lm, pipeline, templates as T
from sftlab.train import SFTConfig

# ----------------------------------------------------------------------------- part A


def build_format_pool(seed: int, n: int, alpaca):
    r = random.Random(seed)
    out = []
    for e in r.sample(alpaca, len(alpaca)):
        c = r.choice(formats.TRAIN)
        if not formats.eligible(e, c):
            continue
        x = formats.make_example(e, c)
        if c.check(x["response"]):
            out.append(x)
        if len(out) == n:
            break
    return out


def build_format_eval(alpaca, seed: int, per: int):
    """per prompts x each training constraint, per prompts with the unseen constraint, per prompts unconstrained."""
    r = random.Random(1000 + seed)
    cands = [e for e in r.sample(alpaca, len(alpaca)) if len(formats.sentences(e["response"])) >= 3 and 20 < len(e["response"]) < 600]
    ev = []
    for k in range(per):
        e = cands[k]
        for c in formats.TRAIN + [formats.HELD_OUT]:
            ev.append({**e, "prompt": e["prompt"] + " " + c.text, "constraint": c.name})
        ev.append({**e, "constraint": "none"})
    return ev


def adherence(replies, ev):
    by = {}
    for r, e in zip(replies, ev):
        if e["constraint"] == "none":
            ok = r["stopped"]
        else:
            ok = r["stopped"] and formats.BY_NAME[e["constraint"]].check(r["reply"].strip())
        by.setdefault(e["constraint"], []).append(ok)
    seen = [v for k, vs in by.items() if k not in ("none", formats.HELD_OUT.name) for v in vs]
    return {"seen": float(np.mean(seen)), "unseen": float(np.mean(by[formats.HELD_OUT.name])), "none_stop": float(np.mean(by["none"])),
            **{k: float(np.mean(v)) for k, v in by.items()}}


def fewshot_prompt(demos, ex):
    s = "".join(f"### Instruction:\n{d['prompt']}\n\n### Response:\n{d['response']}\n\n" for d in demos)
    return s + f"### Instruction:\n{ex['prompt']}\n\n### Response:\n"


@torch.no_grad()
def fewshot_eval(tok, base, demos, ev, max_new=160):
    prompts = [tok(fewshot_prompt(demos, e), add_special_tokens=False)["input_ids"] for e in ev]
    g = lm.greedy(base, tok, prompts, max_new=max_new, stop_ids=[tok.convert_tokens_to_ids("<|endoftext|>")], batch=16)
    for r in g:
        t = r["text"].replace("<|endoftext|>", "")
        cut = t.find("###")
        r["stopped"] = cut >= 0 or r["stopped"]
        r["reply"] = (t[:cut] if cut >= 0 else t).strip()
    return g


def part_a(out, tok, Ns, seeds, per, steps_target, lr):
    out("A. Format adherence against the number of SFT examples (SmolLM2-135M base, Alpaca prompts, 5 trained format constraints)")
    alp = [e for e in data.load_alpaca() if not e["context"] and 3 <= len(e["response"]) <= 600]
    ev = build_format_eval(alp, 0, per)
    out(f"   Evaluation: {per} held-out prompts x (5 trained constraints, 1 unseen constraint, 1 with no constraint) = {len(ev)} prompts, greedy, 160 new tokens.")
    out("   'adheres' = the reply ended with <|im_end|> within the budget AND passes the exact checker for its constraint.")
    out("   'seen' averages the 5 trained constraints; 'unseen' is 'Begin your answer with the word Sure,' (never in training);")
    out("   'stops' is the share of unconstrained prompts whose reply ended with <|im_end|>.")
    out(f"   Training: lr {lr:g}, batch 16, about {steps_target} optimiser steps for every N (so small N gets many epochs), responses rewritten to satisfy the constraint.")
    out("   The step count is held fixed so that N is the only thing that changes. A first attempt with lr 2e-5 and 48 steps gave no stopping at all for N <= 64.")
    out(f"   {'N':>6} {'epochs':>7} {'steps':>6} {'adheres: seen':>17} {'unseen':>9} {'stops (no constraint)':>22} {'seeds':>6}")
    res = {}
    base = lm.load_model(lm.BASE).eval()
    g0 = pipeline.completions(base, tok, ev, max_new=160)
    r0 = adherence(g0, ev)
    out(f"   {0:>6} {'-':>7} {'-':>6} {r0['seen']:17.3f} {r0['unseen']:9.3f} {r0['none_stop']:22.3f} {1:>6}   (base model, chat template)")
    res["0"] = r0
    del base
    for N in Ns:
        ep = max(1, math.ceil(steps_target * 16 / N))
        steps = ep * math.ceil(N / 16)
        rs = []
        for seed in seeds:
            tr = build_format_pool(seed, N, alp)
            enc = pipeline.encode_examples(tok, tr, "assistant", 320)
            model, _ = pipeline.finetune(tok, enc, SFTConfig(lr=lr, epochs=ep, seed=seed))
            rs.append(adherence(pipeline.completions(model, tok, ev, max_new=160), ev))
            del model
            torch.cuda.empty_cache()
        m = lambda k: np.mean([r[k] for r in rs])
        s = lambda k: np.std([r[k] for r in rs])
        out(f"   {N:>6} {ep:>7} {steps:>6} {m('seen'):9.3f}+-{s('seen'):.3f} {m('unseen'):9.3f} {m('none_stop'):22.3f} {len(seeds):>6}")
        res[str(N)] = {k: m(k) for k in rs[0]} | {"seen_std": s("seen")}
    # few-shot prompting of the base model
    out("   Few-shot prompting of the same base model (no weight update). Demos use the plain '### Instruction / ### Response' text format and")
    out("   the reply is cut at the next '###'; the demos are drawn from the same pool and cover each constraint equally.")
    base = lm.load_model(lm.BASE).eval()
    for k_per in [1, 2]:
        rs = []
        for seed in seeds:
            pool = build_format_pool(seed, 400, alp)
            demos = []
            for c in formats.TRAIN:
                demos += [x for x in pool if x["constraint"] == c.name][:k_per]
            random.Random(seed).shuffle(demos)
            rs.append(adherence(fewshot_eval(tok, base, demos, ev), ev))
        m = lambda k: np.mean([r[k] for r in rs])
        out(f"   {'few-shot':>6} {'-':>7} {'-':>6} {m('seen'):9.3f}+-{np.std([r['seen'] for r in rs]):.3f} {m('unseen'):9.3f} {m('none_stop'):22.3f} {len(seeds):>6}   ({5 * k_per} demonstrations in the prompt)")
        res[f"fewshot_{5 * k_per}"] = {k: m(k) for k in rs[0]}
    out("")
    return res


# ----------------------------------------------------------------------------- part B
NORM = re.compile(r"[^a-z0-9 ]")


def norm(s):
    return " ".join(NORM.sub(" ", s.lower()).split())


def correct(reply, aliases):
    r = norm(reply)
    return any(norm(a) and norm(a) in r for a in aliases)


def load_trivia(n):
    from datasets import load_dataset
    d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    items = []
    for r in d:
        a = r["answer"]
        if len(a["value"]) < 40:
            items.append({"prompt": r["question"], "response": a["value"], "aliases": a["aliases"] + [a["value"]]})
        if len(items) >= n:
            break
    return items


@torch.no_grad()
def classify(model, tok, items, shots):
    fs = "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in shots)
    prompts = [tok(fs + f"Question: {e['prompt']}\nAnswer:", add_special_tokens=False)["input_ids"] for e in items]
    g = lm.greedy(model, tok, prompts, max_new=10, stop_ids=[tok.convert_tokens_to_ids("<|endoftext|>")], batch=64)
    return [correct(r["text"].split("\n")[0], e["aliases"]) for r, e in zip(g, items)]


@torch.no_grad()
def accuracy(model, tok, items):
    if not items:
        return float("nan")
    g = pipeline.completions(model, tok, items, max_new=12, batch=100)
    return float(np.mean([correct(r["reply"], e["aliases"]) for r, e in zip(g, items)]))


def part_b(out, tok, repo, n_pool, n_train, epochs, mixes, seeds, n_dev):
    out(f"B. Known against unknown answers ({repo}, TriviaQA validation questions, closed book)")
    items = load_trivia(n_pool)
    rng = random.Random(0)
    shots_pool, items = items[:5], items[5:]
    shots = [(e["prompt"], e["response"]) for e in shots_pool]
    model0 = lm.load_model(repo).eval()
    known = classify(model0, tok, items, shots)
    out(f"   {len(items)} questions; the base model answers {sum(known)} ({sum(known) / len(items):.1%}) correctly with a 5-shot prompt (answer contains an accepted alias).")
    K = [e for e, k in zip(items, known) if k]
    U = [e for e, k in zip(items, known) if not k]
    rng.shuffle(K)
    rng.shuffle(U)
    dev_K, dev_U = K[:n_dev // 2], U[:n_dev // 2]
    K, U = K[n_dev // 2:], U[n_dev // 2:]
    out(f"   Dev set: {len(dev_K)} known + {len(dev_U)} unknown questions, never trained on. Training pool: {len(K)} known, {len(U)} unknown.")
    out(f"   Each run trains on {n_train} questions with the given share of UNKNOWN answers, {epochs} epochs, lr 2e-5, batch 16, chat template.")
    out("   Columns: accuracy (answer contains an accepted alias, greedy) on the training questions that were known / unknown, then on the two dev sets.")
    res = {}
    for share in mixes:
        for seed in seeds:
            r = random.Random(seed)
            nu = int(round(share * n_train))
            tr_U = r.sample(U, nu)
            tr_K = r.sample(K, n_train - nu)
            tr = tr_U + tr_K
            enc = pipeline.encode_examples(tok, tr, "assistant", 128)
            evK = tr_K[:150]
            evU = tr_U[:150]

            def on_epoch(ep, model):
                return {"train_known": accuracy(model, tok, evK), "train_unknown": accuracy(model, tok, evU),
                        "dev_known": accuracy(model, tok, dev_K), "dev_unknown": accuracy(model, tok, dev_U)}
            model, log = pipeline.finetune(tok, enc, SFTConfig(lr=2e-5, epochs=epochs, seed=seed), repo=repo, on_epoch=on_epoch)
            res[f"{share}_{seed}"] = log
            del model
            torch.cuda.empty_cache()
        out(f"   unknown share {share:.0%} ({len(seeds)} seed{'s' if len(seeds) > 1 else ''}, mean):")
        out(f"     {'epoch':>6} {'train loss':>11} {'fit: known':>11} {'fit: unknown':>13} {'dev: known':>11} {'dev: unknown':>13}")
        for ep in range(epochs):
            rows = [res[f"{share}_{s}"][ep] for s in seeds]
            f = lambda k: np.nanmean([x[k] for x in rows])
            out(f"     {ep + 1:>6} {f('train_loss'):11.3f} {f('train_known'):11.3f} {f('train_unknown'):13.3f} {f('dev_known'):11.3f} {f('dev_unknown'):13.3f}")
    base_acc = (accuracy(model0, tok, dev_K[:0]) if False else None)
    out("   Reference: before fine-tuning the dev-known set is answered correctly by construction (that is how it was chosen, with the 5-shot prompt),")
    out("   so a drop in 'dev: known' is damage caused by fine-tuning (or by the change of prompt format; compare the 0% row).")
    out("")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--Ns", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--per", type=int, default=40)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--b-repo", default=lm.BASE_360)
    ap.add_argument("--b-pool", type=int, default=6000)
    ap.add_argument("--b-train", type=int, default=400)
    ap.add_argument("--b-epochs", type=int, default=8)
    ap.add_argument("--b-dev", type=int, default=300)
    ap.add_argument("--b-seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    out = _common.Report("06_what_sft_teaches" if a.part == "all" else f"06_what_sft_teaches_{a.part}")
    tok = lm.chat_tokenizer()
    res = {}
    if a.part in ("a", "all"):
        res["format"] = part_a(out, tok, a.Ns, a.seeds, a.per, a.steps, a.lr)
    if a.part in ("b", "all"):
        res["knowledge"] = part_b(out, tok, a.b_repo, a.b_pool, a.b_train, a.b_epochs, [0.0, 0.5, 1.0], a.b_seeds, a.b_dev)
    out.save()
    _common.save_json("06_what_sft_teaches" if a.part == "all" else f"06_what_sft_teaches_{a.part}", res)


if __name__ == "__main__":
    main()
