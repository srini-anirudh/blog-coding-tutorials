"""Section 10: how much SFT? Learning rate, epochs, the early signs of over-training, truncation, verbosity.

SmolLM2-135M (base) on Alpaca. Every row is one training run unless a column says otherwise (seeds listed).
A. Learning rate. B. Epochs on a small set: which signal moves first? C. Truncated training data and the stop token.
D. Does verbose data make a verbose model? E. Is a post-SFT drop on a benchmark a format problem?
F. A template mismatch between training and serving.
"""

import argparse
import random
import re

import _common
import numpy as np
import torch
from sftlab import data, lm, metrics, pipeline, templates as T
from sftlab.train import SFTConfig, train, mean_nll


def alpaca_pool():
    return [e for e in data.load_alpaca() if not e["context"] and 10 < len(e["response"]) < 1200]


def fmt(rows, keys):
    return "".join(f"{np.mean([r[k] for r in rows]):10.3f}" for k in keys)


def part_a(out, tok, seeds, lrs, N, wiki):
    out(f"A. Learning rate. {N} Alpaca examples, 3 epochs, batch 16, cosine decay. Validation: 200 held-out examples (assistant tokens).")
    out(f"   'wiki' = NLL on 100 WikiText paragraphs, a probe of what moved elsewhere. 'stop' = share of replies that ended with <|im_end|> within 128 tokens.")
    pool = alpaca_pool()
    keys = ["val_nll", "wiki_nll", "stop", "len", "distinct2"]
    out(f"   {'lr':>8} {'val NLL':>10} {'wiki NLL':>10} {'stop':>10} {'mean len':>10} {'distinct-2':>10}   seeds")
    base = lm.load_model(lm.BASE).eval()
    tr0, va0 = data.split(pool, N, 200, seed=0)
    ve0 = pipeline.encode_examples(tok, va0, "assistant", 384)
    r0 = pipeline.behaviour_eval(base, tok, ve0, va0, wiki)
    out(f"   {'none':>8} " + fmt([r0], keys) + "   (base model, chat template)")
    del base
    res = {"none": r0}
    for lr in lrs:
        rows = []
        for seed in seeds:
            tr, va = data.split(pool, N, 200, seed=seed)
            enc = pipeline.encode_examples(tok, tr, "assistant", 384)
            ve = pipeline.encode_examples(tok, va, "assistant", 384)
            model, _ = pipeline.finetune(tok, enc, SFTConfig(lr=lr, epochs=3, seed=seed))
            rows.append(pipeline.behaviour_eval(model, tok, ve, va, wiki, seed=seed))
            del model
            torch.cuda.empty_cache()
        out(f"   {lr:8.0e} " + fmt(rows, keys) + f"   {len(seeds)}")
        res[str(lr)] = rows
    out("")
    return res


def part_b(out, tok, seeds, N, epochs_at, wiki):
    out(f"B. Epochs on a small set. {N} Alpaca examples, constant lr 2e-5 (so the checkpoints are comparable), batch 16, checkpoints at the listed epochs.")
    out("   'train NLL' and 'val NLL' are assistant-token NLL. 'regurgitate' = share of greedy replies on held-out prompts whose word set has Jaccard >= 0.8")
    out("   with SOME training response. 'distinct-2' = distinct bigram share among 8 sampled replies to the same prompt (T=1), averaged over 20 prompts.")
    pool = alpaca_pool()
    keys = ["train_nll", "val_nll", "wiki_nll", "regurgitate", "distinct2", "len", "stop"]
    allrows = {}
    for seed in seeds:
        tr, va = data.split(pool, N, 200, seed=seed)
        enc = pipeline.encode_examples(tok, tr, "assistant", 384)
        ve = pipeline.encode_examples(tok, va, "assistant", 384)
        resp = [e["response"] for e in tr]

        def on_epoch(ep, model):
            if ep not in epochs_at:
                return None
            r = pipeline.behaviour_eval(model, tok, ve, va, wiki, resp, seed=seed)
            r["train_nll"] = mean_nll(model, enc, tok.pad_token_id)["token_nll"]
            return r
        model, log = pipeline.finetune(tok, enc, SFTConfig(lr=2e-5, epochs=max(epochs_at), seed=seed, min_lr_frac=1.0, warmup_frac=0.03), on_epoch=on_epoch)
        allrows[seed] = [x for x in log if x["epoch"] in epochs_at]
        del model
        torch.cuda.empty_cache()
    out(f"   {'epochs':>7} " + "".join(f"{k:>12}" for k in keys) + f"   (mean of {len(seeds)} seed{'s' if len(seeds) > 1 else ''})")
    for i, ep in enumerate(epochs_at):
        rows = [allrows[s][i] for s in seeds]
        out(f"   {ep:>7} " + "".join(f"{np.mean([r[k] for r in rows]):12.3f}" for k in keys))
    best = min(range(len(epochs_at)), key=lambda i: np.mean([allrows[s][i]["val_nll"] for s in seeds]))
    out(f"   Validation NLL is lowest at epoch {epochs_at[best]}.")
    out("")
    return allrows


def part_c(out, tok, seeds, N):
    out(f"C. Truncated training data. {N} Alpaca examples tokenised with max length L, 3 epochs, lr 2e-5. Evaluation: 100 held-out prompts, 128 new tokens.")
    pool = alpaca_pool()
    out(f"   {'L':>6} {'examples cut':>13} {'cut ones that keep <|im_end|>':>31} {'stop rate':>10} {'mean len':>9}   seeds")
    res = {}
    for L in [48, 96, 384]:
        rows = []
        for seed in seeds:
            tr, va = data.split(pool, N, 100, seed=seed)
            enc = pipeline.encode_examples(tok, tr, "assistant", L)
            cut = [e for e in enc if e.truncated]
            keep = sum(T.has_turn_end(e, tok) for e in cut)
            model, _ = pipeline.finetune(tok, enc, SFTConfig(lr=2e-5, epochs=3, seed=seed))
            g = pipeline.completions(model, tok, va, max_new=128)
            rows.append(dict(cut=len(cut) / len(enc), keep=keep / max(1, len(cut)), stop=np.mean([x["stopped"] for x in g]), len=np.mean([len(x["ids"]) for x in g])))
            del model
            torch.cuda.empty_cache()
        out(f"   {L:>6} {np.mean([r['cut'] for r in rows]):13.2f} {np.mean([r['keep'] for r in rows]):31.3f} {np.mean([r['stop'] for r in rows]):10.3f} {np.mean([r['len'] for r in rows]):9.1f}   {len(seeds)}")
        res[L] = rows
    out("   Training on cut examples loses the stop token for most of the data, and the model learns to keep going. At L = 48 nearly every")
    out("   example is cut; at L = 384 few are.")
    out("")
    return res


def part_d(out, tok, seeds, N):
    out(f"D. Verbose data gives a verbose model. {N} examples from the shorter or the longer half of Alpaca responses (by tokens), lr 2e-5, 3 epochs.")
    pool = alpaca_pool()
    lens = {id(e): len(tok(e["response"], add_special_tokens=False)["input_ids"]) for e in pool}
    med = np.median(list(lens.values()))
    short = [e for e in pool if lens[id(e)] <= med]
    long = [e for e in pool if lens[id(e)] > med]
    out(f"   median response = {med:.0f} tokens. Validation prompts are drawn from the whole pool; 'mean len' is the generated length in tokens (budget 160).")
    out(f"   {'training responses':<24} {'mean train len':>15} {'mean len':>9} {'stop rate':>10}   seeds")
    res = {}
    for name, sub in [("shorter half", short), ("longer half", long)]:
        rows = []
        for seed in seeds:
            tr, _ = data.split(sub, N, 0, seed=seed)
            _, va = data.split(pool, 0, 100, seed=seed)
            enc = pipeline.encode_examples(tok, tr, "assistant", 384)
            model, _ = pipeline.finetune(tok, enc, SFTConfig(lr=2e-5, epochs=3, seed=seed))
            g = pipeline.completions(model, tok, va, max_new=160)
            rows.append(dict(train_len=np.mean([lens[id(e)] for e in tr]), len=np.mean([len(x["ids"]) for x in g]), stop=np.mean([x["stopped"] for x in g])))
            del model
            torch.cuda.empty_cache()
        out(f"   {name:<24} {np.mean([r['train_len'] for r in rows]):15.1f} {np.mean([r['len'] for r in rows]):9.1f} {np.mean([r['stop'] for r in rows]):10.3f}   {len(seeds)}")
        res[name] = rows
    out("")
    return res


def load_arc(n):
    from datasets import load_dataset
    d = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    items = [r for r in d if len(r["choices"]["label"]) == 4 and r["answerKey"] in "ABCD"]
    return random.Random(0).sample(items, n)


@torch.no_grad()
def arc_loglik(model, tok, items):
    """Format-free: pick the choice with the highest mean log-probability after 'Answer:'."""
    ok = 0
    for r in items:
        ctx = tok(f"Question: {r['question']}\nAnswer:", add_special_tokens=False)["input_ids"]
        best, bi = -1e9, -1
        for i, c in enumerate(r["choices"]["text"]):
            cid = tok(" " + c, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([ctx + cid], device=model.device)
            lp = torch.log_softmax(model(input_ids=ids).logits[0, len(ctx) - 1:-1].float(), -1)
            s = float(lp.gather(1, torch.tensor(cid, device=model.device)[:, None]).mean())
            if s > best:
                best, bi = s, i
        ok += "ABCD"[bi] == r["answerKey"]
    return ok / len(items)


def arc_prompt(r, chat):
    body = f"Question: {r['question']}\n" + "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
    return body + ("\nAnswer with the letter only." if chat else "\nAnswer:")


@torch.no_grad()
def arc_generate(model, tok, items, chat):
    ps = [T.encode_prompt(tok, [{"role": "user", "content": arc_prompt(r, True)}]) if chat else tok(arc_prompt(r, False), add_special_tokens=False)["input_ids"] for r in items]
    g = lm.greedy(model, tok, ps, max_new=24, stop_ids=[tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")], batch=50)
    strict = lenient = 0
    for r, x in zip(items, g):
        t = x["text"].replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        strict += bool(re.fullmatch(rf"{r['answerKey']}[.)]?", t))
        gold_text = r["choices"]["text"][r["choices"]["label"].index(r["answerKey"])].lower()
        lenient += bool(re.search(rf"(^|\W){r['answerKey']}(\W|$)", t)) or gold_text in t.lower()
    return strict / len(items), lenient / len(items)


def part_e(out, tok, seeds, N, n_arc):
    out(f"E. Is a post-SFT drop a format problem? {n_arc} ARC-Easy questions (4 choices, chance 0.25). SFT: {N} Alpaca examples, lr 2e-5, 3 epochs.")
    items = load_arc(n_arc)
    pool = alpaca_pool()
    base = lm.load_model(lm.BASE).eval()
    ll_base = arc_loglik(base, tok, items)
    s_base, l_base = arc_generate(base, tok, items, chat=False)
    out(f"   {'model':<28} {'log-likelihood acc':>19} {'generated letter, strict':>25} {'generated, lenient':>20}")
    out(f"   {'base (plain prompt)':<28} {ll_base:19.3f} {s_base:25.3f} {l_base:20.3f}")
    del base
    rows = []
    for seed in seeds:
        tr, _ = data.split(pool, N, 0, seed=seed)
        enc = pipeline.encode_examples(tok, tr, "assistant", 384)
        model, _ = pipeline.finetune(tok, enc, SFTConfig(lr=2e-5, epochs=3, seed=seed))
        ll = arc_loglik(model, tok, items)
        s, l = arc_generate(model, tok, items, chat=True)
        rows.append((ll, s, l))
        del model
        torch.cuda.empty_cache()
    a = np.mean(rows, 0)
    out(f"   {'after SFT (chat template)':<28} {a[0]:19.3f} {a[1]:25.3f} {a[2]:20.3f}   ({len(seeds)} seeds)")
    out("   Strict = the whole reply is the right letter. Lenient = the right letter stands alone, or the right choice text appears in the reply.")
    out("")
    return {"base": [ll_base, s_base, l_base], "sft": rows}


def part_f(out, tok, seeds, N):
    out(f"F. Train/serve template mismatch on our own model. SFT on {N} Alpaca examples with the right template, then serve with small mistakes (100 held-out prompts, greedy, 96 new tokens).")
    pool = alpaca_pool()
    allres = []
    for seed in seeds:
        tr, _ = data.split(pool, N, 0, seed=seed)
        va = [e for e in data.split(pool, 0, 400, seed=seed)[1] if len(e["response"]) < 400][:100]
        model, _ = pipeline.finetune(tok, pipeline.encode_examples(tok, tr, "assistant", 384), SFTConfig(lr=2e-5, epochs=3, seed=seed))
        allres.append(pipeline.serve_eval(model, tok, va))
        del model
        torch.cuda.empty_cache()
    mean = {name: {k: float(np.mean([r[name][k] for r in allres])) for k in allres[0][name]} for name in allres[0]}
    out(f"   mean of {len(seeds)} seed{'s' if len(seeds) > 1 else ''}:")
    pipeline.print_serve(out, mean)
    out("")
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="abcdef")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--n-arc", type=int, default=300)
    a = ap.parse_args()
    out = _common.Report("10_how_much_sft")
    tok = lm.chat_tokenizer()
    wiki = pipeline.wiki_encoded(tok)
    res = {}
    if "a" in a.parts:
        res["lr"] = part_a(out, tok, a.seeds[:1], [1e-6, 5e-6, 2e-5, 1e-4, 5e-4], a.N, wiki)
    if "b" in a.parts:
        res["epochs"] = part_b(out, tok, a.seeds, 200, [1, 3, 6, 10, 20, 30], wiki)
    if "c" in a.parts:
        res["truncation"] = part_c(out, tok, a.seeds, 500)
    if "d" in a.parts:
        res["verbosity"] = part_d(out, tok, a.seeds, 500)
    if "e" in a.parts:
        res["arc"] = part_e(out, tok, a.seeds, 1000, a.n_arc)
    if "f" in a.parts:
        res["serve_mismatch"] = part_f(out, tok, a.seeds, a.N)
    out.save()
    _common.save_json("10_how_much_sft", res)


if __name__ == "__main__":
    main()
