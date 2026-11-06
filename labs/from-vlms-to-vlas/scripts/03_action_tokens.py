"""Action tokenizers: 256 bins, the information each bin token carries, and FAST.

Part A bins 7-D actions as RT-1, RT-2 and OpenVLA do and measures the rounding error. With --vocab it also picks the
256 least used ids of the SmolLM2 tokenizer on WikiText-2 and writes one action with them, as RT-2 and OpenVLA do.
Part B measures H(b_t | b_{t-1}) for the same motions sampled at 5 to 200 Hz.
Part C trains a FAST tokenizer (DCT, rounding, BPE) on one-second chunks at 50 Hz and counts tokens per chunk.
The actions are synthetic smooth curves, not robot data.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlalab.report import Report  # noqa: E402
from vlalab.tokenizers import (ActionTokenCodec, FASTTokenizer, UniformBinner,  # noqa: E402
                               conditional_entropy_bits, least_used_token_ids, smooth_actions)


def part_a(rep, vocab: bool):
    acts = smooth_actions(500, 2.0, 50, seed=0)
    binner = UniformBinner(256).fit(acts)
    test = smooth_actions(200, 2.0, 50, seed=1)
    inside = (test >= binner.low) & (test <= binner.high)
    err = np.abs(binner.decode(binner.encode(test)) - test)[inside]
    rel = (err / np.broadcast_to(binner.high - binner.low, test.shape)[inside])
    rep.line("Part A. 256 uniform bins, range from the 1st to 99th percentile of 500 training episodes.")
    rep.line(f"Test values inside the range: {inside.mean():.4f}. Largest rounding error inside it: "
             f"{rel.max():.5f} of the range (half a bin is 1/512 = {1 / 512:.5f}); mean {rel.mean():.5f}.")
    out = {"inside_range": float(inside.mean()), "max_err_frac_of_range": float(rel.max()),
           "mean_err_frac_of_range": float(rel.mean())}
    if vocab:
        out["vocab"] = vocab_demo(rep, binner, test[0, 0])
    return out


def vocab_demo(rep, binner, action):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    text = list(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
    counts = Counter()
    for ids in tok(text, add_special_tokens=False)["input_ids"]:
        counts.update(ids)
    v = len(tok)
    action_ids = least_used_token_ids(counts, v, 256, protected=set(tok.all_special_ids))
    codec = ActionTokenCodec(binner, action_ids)
    ids = codec.encode(action)
    unseen = sum(counts.get(i, 0) == 0 for i in action_ids)
    rep.line(f"Vocabulary demo: SmolLM2 tokenizer ({v} ids), counts over WikiText-2 train "
             f"({sum(counts.values())} tokens). {unseen} of the 256 chosen ids never occur there.")
    rep.line(f"Action {np.round(action, 3).tolist()}")
    rep.line(f"  -> bins {binner.encode(action).tolist()}")
    rep.line(f"  -> ids {ids}")
    rep.line(f"  -> as text {tok.convert_ids_to_tokens(ids)}")
    back = codec.decode(ids)
    rep.line(f"  -> decoded {np.round(back, 3).tolist()}")
    return {"vocab_size": v, "corpus_tokens": sum(counts.values()), "chosen_ids_unseen": int(unseen),
            "example_ids": ids}


def part_b(rep):
    rep.line("\nPart B. Information per bin token as the control rate rises. 300 episodes of 4 s, 6 arm dimensions.")
    rows = []
    for noise in (0.0, 0.01):
        for hz in (5, 10, 25, 50, 100, 200):
            acts = smooth_actions(300, 4.0, hz, seed=2, noise=noise)[..., :6]
            bins = UniformBinner(256).fit(acts).encode(acts)
            seqs = bins.transpose(0, 2, 1).reshape(-1, bins.shape[1])  # one sequence per episode and dimension
            h, hc = conditional_entropy_bits(seqs)
            rows.append({"jitter": noise, "Hz": hz, "tokens per s (7-D)": 7 * hz, "H(b) bits": round(h, 2),
                         "H(b_t | b_t-1) bits": round(hc, 2), "bits per second, one dim": round(hc * hz, 1)})
    rep.table(rows)
    rep.line("jitter = std of white noise added to each step, in units where each curve spans a few units.")
    return rows


def part_c(rep, n_train: int):
    rep.line(f"\nPart C. FAST on one-second chunks at 50 Hz, 7-D (50 x 7 = 350 bin tokens). "
             f"BPE vocabulary 1024, trained on {n_train} chunks; 300 test chunks.")
    train = smooth_actions(n_train, 1.0, 50, seed=3)
    test = smooth_actions(300, 1.0, 50, seed=4)
    rows = []
    for scale in (5.0, 10.0, 20.0):
        t0 = time.time()
        fast = FASTTokenizer(scale=scale, vocab_size=1024).fit(train)
        fit_s = time.time() - t0
        coef = np.stack([fast.coefficients(c) for c in test])
        lengths = [len(fast.encode(c)) for c in test]
        recon = np.stack([fast.decode(fast.encode(c)) for c in test])
        rng = fast.binner.high - fast.binner.low
        rmse = np.sqrt((((recon - test) / rng) ** 2)[..., :6].mean())
        clipped = np.clip(test, fast.binner.low, fast.binner.high)  # error from rounding alone
        rmse_round = np.sqrt((((recon - clipped) / rng) ** 2)[..., :6].mean())
        rows.append({"scale": scale, "nonzero coefs / 350": round(float((coef != 0).sum((1, 2)).mean()), 1),
                     "FAST tokens mean": round(float(np.mean(lengths)), 1),
                     "min-max": f"{min(lengths)}-{max(lengths)}", "compression vs 350": round(350 / np.mean(lengths), 1),
                     "arm RMSE / range": round(float(rmse), 4),
                     "rounding only": round(float(rmse_round), 4), "BPE fit s": round(fit_s, 1)})
    rep.table(rows)
    rep.line("RMSE is over the 6 arm dimensions, as a fraction of the 1st-99th percentile range. 'rounding only' "
             "compares with the target clipped to that range, which FAST does before the DCT.")
    rep.line("The FAST paper reports 30 to 60 tokens per 7-D one-second chunk on real robot data. These curves are "
             "synthetic, so the counts are not comparable one to one.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", action="store_true", help="tokenizer demo; needs SmolLM2 and WikiText-2 from the Hub")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "03_action_tokens"))
    args = ap.parse_args()
    rep = Report(args.out, " ".join(["scripts/03_action_tokens.py"] + sys.argv[1:]))
    res = {"a": part_a(rep, args.vocab), "b": part_b(rep), "c": part_c(rep, args.n_train)}
    rep.save_json(res)


if __name__ == "__main__":
    main()
