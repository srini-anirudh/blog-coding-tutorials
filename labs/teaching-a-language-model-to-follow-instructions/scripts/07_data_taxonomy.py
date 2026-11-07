"""Sections 7 and 8: the instruction-data taxonomy and how the data gets built. CPU only.

A. What the two public datasets contain, by category and length (this sets up the regimes of script 05).
B. Checkers: which categories can be verified exactly, and a case where a reference-based metric cannot verify.
C. Capped-proportional mixing against proportional mixing.
D. Contamination: n-gram overlap between instruction data and GSM8K test questions, and what a paraphrase does to it.
E. Near-duplicates in a Self-Instruct-style set (Alpaca instructions), by ROUGE-L.
"""

import argparse
import random
import re
from collections import Counter, defaultdict

import _common
import numpy as np
from datasets import load_dataset
from sftlab import data, lm, metrics, verifiers as V


def part_a(out, tok):
    out("A. Dolly (human-written) by category, and Alpaca (teacher-written); token counts with the SmolLM2 tokenizer")
    dol = data.load_dolly()
    alp = data.load_alpaca()
    by = defaultdict(list)
    for e in dol:
        by[e["category"]].append(e)
    out(f"   {'set':<26} {'n':>6} {'instr+context tokens (median)':>30} {'response tokens (median)':>25} {'ratio':>6}")
    res = {}
    rows = [(k, v) for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))] + [("alpaca (all)", alp)]
    for k, v in rows:
        sub = random.Random(0).sample(v, min(len(v), 1500))
        pi = [len(tok(e["prompt"], add_special_tokens=False)["input_ids"]) for e in sub]
        ro = [len(tok(e["response"], add_special_tokens=False)["input_ids"]) for e in sub]
        out(f"   {'dolly: ' + k if k != 'alpaca (all)' else k:<26} {len(v):6d} {np.median(pi):30.0f} {np.median(ro):25.0f} {np.median(pi) / np.median(ro):6.2f}")
        res[k] = dict(n=len(v), prompt_med=float(np.median(pi)), resp_med=float(np.median(ro)))
    out("")
    return res


def part_b(out):
    out("B. Checkers")
    rows = []
    rows.append(("structured output", "JSON parses, exact keys and types", V.check_json('{"name": "Ada", "age": 36}', {"name": str, "age": int}), V.check_json('{"name": "Ada", "age": "36"}', {"name": str, "age": int})))
    rows.append(("classification", "reply equals the gold label", V.check_label("positive", {"positive", "negative"}, "positive"), V.check_label("The sentiment is positive.", {"positive", "negative"}, "positive")))
    rows.append(("mathematics", "last number equals the gold answer", V.check_math("... so the total is $1,250.", 1250.0), V.check_math("... so the total is 1,205.", 1250.0)))
    rows.append(("coding", "run the candidate with asserts", V.check_code("def add(a, b):\n    return a + b", "assert add(2, 3) == 5"), V.check_code("def add(a, b):\n    return a - b", "assert add(2, 3) == 5")))
    out(f"   {'category':<20} {'checker':<38} {'accepts a correct reply':>24} {'rejects a wrong reply':>22}")
    for cat, desc, good, bad in rows:
        out(f"   {cat:<20} {desc:<38} {str(good):>24} {str(not bad):>22}")
    out("   Each checker is exact for its category. None of them needs a human or a teacher model.")
    # reference-based metric on summaries
    dol = [e for e in data.load_dolly() if e["category"] == "summarization" and 15 < len(e["response"].split()) < 80]
    rng = random.Random(0)

    def corrupt(s):
        s = re.sub(r"\d+", lambda m: str(int(m.group()) + rng.randint(2, 9)), s)
        return re.sub(r"\b(is|was|are|were|has|have|can)\b", r"\1 not", s, count=2)
    sub = rng.sample(dol, min(300, len(dol)))
    f_bad = [metrics.f1(corrupt(e["response"]), e["response"]) for e in sub]
    f_other = [metrics.f1(sub[(i + 1) % len(sub)]["response"], e["response"]) for i, e in enumerate(sub)]
    out(f"   Reference-based metric on {len(sub)} Dolly summaries: token F1 of the reference against a copy with the numbers changed and negations inserted")
    out(f"   = {np.mean(f_bad):.3f} (a plainly wrong summary), against {np.mean(f_other):.3f} for another example's summary. Overlap cannot see the error.")
    out("")


def part_c(out):
    out("C. Mixing weights from source sizes. Sizes are the public sets this lab loads, plus two sizes quoted for illustration")
    sizes = {"alpaca-cleaned": 51760, "dolly-15k": 15011, "gsm8k-train": 7473, "large-synthetic-set (illustrative)": 900_000, "small-human-set (illustrative)": 1000}
    for cap in [None, 20_000]:
        w = V.proportional(sizes) if cap is None else V.capped_proportional(sizes, cap)
        out(f"   {'proportional' if cap is None else f'capped at {cap:,} per source'}:")
        out("     " + ", ".join(f"{k.split(' (')[0]} {v:.3f}" for k, v in w.items()))
    out("   Proportional mixing gives the illustrative 900k-example source 92% of the mix and the 1,000-example source 0.1%. A cap")
    out("   restores a spread without a hand-tuned weight. Whether the capped weights are good is a mixture question (another post).")
    out("")


def part_d(out):
    out("D. Contamination: word n-gram overlap of GSM8K test questions with the instruction data (instructions plus responses)")
    gsm = load_dataset("openai/gsm8k", "main", split="test")["question"]
    train = [e["prompt"] + " " + e["response"] for e in data.load_alpaca()] + [e["prompt"] + " " + e["response"] for e in data.load_dolly()]
    out(f"   {len(gsm)} GSM8K test questions against {len(train)} Alpaca + Dolly examples")
    res = {}
    for n in [13, 8, 5]:
        frac, hit = V.contaminated_fraction(gsm, train, n)
        res[n] = (frac, len(hit))
        out(f"   n = {n:2d}: {frac:.4f} of test questions share an n-gram ({len(hit)} questions)")
    out("   A planted copy and a light paraphrase of 100 test questions (every 5th word replaced by a different word):")
    rng = random.Random(0)
    sub = rng.sample(list(gsm), 100)

    def para(q):
        w = q.split()
        return " ".join("thing" if i % 5 == 4 else x for i, x in enumerate(w))
    planted = [q for q in sub]
    pr = [para(q) for q in sub]
    for label, texts in [("exact copy", planted), ("light paraphrase", pr)]:
        row = []
        for n in [13, 8, 5]:
            hits = sum(bool(metrics.ngrams(t, n) & set().union(*[metrics.ngrams(q, n) for q in sub])) for t in texts)
            row.append(f"n={n}: {hits}/100")
        rl = np.mean([V.rouge_l(t, q) for t, q in zip(texts, sub)])
        out(f"   {label:<17} detected at " + ", ".join(row) + f";  mean ROUGE-L against the original {rl:.2f}")
    out("   The 13-gram check catches copies and misses the paraphrase that keeps most words. That is the gap the post says embedding-level")
    out("   checks must cover; those are not built here.")
    out("")
    return res


def part_e(out, n):
    out(f"E. Near-duplicate instructions in Alpaca (a Self-Instruct-style set): {n} random instructions, ROUGE-L against earlier ones")
    alp = data.load_alpaca()
    sub = [e["instruction"] for e in random.Random(0).sample(alp, n)]
    index = defaultdict(list)
    kept, dropped = [], 0
    worst = []
    for i, s in enumerate(sub):
        toks = metrics.words(s)
        bigrams = {tuple(toks[j:j + 2]) for j in range(len(toks) - 1)} or {tuple(toks)}
        cand = Counter()
        for b in bigrams:
            for c in index[b]:
                cand[c] += 1
        best = 0.0
        for c, _ in cand.most_common(30):
            best = max(best, V.rouge_l(s, sub[c]))
        worst.append(best)
        if best >= 0.7:
            dropped += 1
        else:
            kept.append(i)
        for b in bigrams:
            index[b].append(i)
    w = np.array(worst)
    out(f"   share with ROUGE-L >= 0.7 to some earlier instruction (the Self-Instruct filter threshold): {dropped / n:.3f}")
    out(f"   share >= 0.5: {np.mean(w >= 0.5):.3f}; share >= 0.3: {np.mean(w >= 0.3):.3f}; median best-match ROUGE-L {np.median(w):.2f}")
    out("   (Only the 30 candidates sharing the most bigrams are scored, so these are lower bounds.)")
    out("")
    return dropped / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dup", type=int, default=5000)
    a = ap.parse_args()
    out = _common.Report("07_data_taxonomy", "cpu")
    tok = lm.chat_tokenizer()
    r = part_a(out, tok)
    part_b(out)
    part_c(out)
    d = part_d(out)
    e = part_e(out, a.n_dup)
    out.save()
    _common.save_json("07_data_taxonomy", {"lengths": r, "gsm8k_overlap": {str(k): v for k, v in d.items()}, "alpaca_dup_rougeL_0.7": e}, "cpu")


if __name__ == "__main__":
    main()
