"""Section 2: same loss, different distribution.

A. Cross-entropy = H(D) + KL(D || p), checked on random distributions and by sampling.
B. Forward KL is mode-covering, reverse KL is mode-seeking (a Gaussian fitted to two bumps).
C. Forward KL barely charges for mass on things the data never shows; it charges a lot for missing mass.
D. The same loss on web text and on chat text, for a base and an instruct model.
E. SFT on a 80/20 mixture of two styles reproduces the mixture: it has no way to say "and not that".
"""

import argparse

import _common
import numpy as np
import torch
from sftlab import data, divergence, lm, pipeline, templates as T
from sftlab.train import SFTConfig, mean_nll


def part_a(out):
    out("A. Cross-entropy = entropy + KL, 6 random distributions over 50 symbols")
    rng = np.random.default_rng(0)
    out(f"   {'H(D)':>8} {'KL(D||p)':>10} {'sum':>8} {'cross-entropy':>14} {'Monte Carlo (1e6 draws)':>24}")
    for _ in range(6):
        d, p = rng.dirichlet(np.ones(50) * 0.5), rng.dirichlet(np.ones(50) * 0.5)
        H, KL, CE = divergence.entropy(d), divergence.kl(d, p), divergence.cross_entropy(d, p)
        x = rng.choice(50, size=1_000_000, p=d)
        mc = float(-np.log(p[x]).mean())
        out(f"   {H:8.4f} {KL:10.4f} {H + KL:8.4f} {CE:14.4f} {mc:24.4f}")
    out("")


def part_b(out):
    out("B. Fit one Gaussian q to a two-bump target D = 0.72 N(-3, 0.6) + 0.28 N(3, 0.6), by grid search")
    x = np.linspace(-12, 12, 4801)
    D = divergence.mixture_pdf(x, [0.72, 0.28], [-3, 3], [0.6, 0.6])
    for name, which in [("forward KL(D||q)  [what SFT minimises]", "forward"), ("reverse KL(q||D)  [what RLHF adds]", "reverse")]:
        mu, sd, val = divergence.fit_gaussian(x, D, which)
        mass = lambda lo, hi: float(np.trapezoid(divergence.normal_pdf(x, mu, sd)[(x >= lo) & (x <= hi)], x[(x >= lo) & (x <= hi)]))
        out(f"   {name}: mu={mu:6.2f} sd={sd:5.2f}  KL={val:7.3f}  q mass near left bump (-5,-1)={mass(-5, -1):.2f}, "
            f"near right bump (1,5)={mass(1, 5):.2f}, between (-1,1)={mass(-1, 1):.2f}")
    out("   Target mass: left bump 0.72, right bump 0.28, between 0.00.")
    out("")


def part_c(out):
    out("C. Which mistakes does each direction charge for? D has 3 behaviours (0.6, 0.3, 0.1), p varies")
    D = np.array([0.6, 0.3, 0.1, 0.0])
    cases = {
        "p = D (reference)": np.array([0.6, 0.3, 0.1, 0.0]),
        "p puts 0.20 on a behaviour D never shows": np.array([0.48, 0.24, 0.08, 0.20]),
        "p puts 0.01 on a behaviour D shows 10%": np.array([0.6, 0.3, 0.01, 0.09]),
        "p drops the 10% behaviour entirely": np.array([0.667, 0.333, 0.0, 0.0]),
    }
    out(f"   {'p':<44} {'KL(D||p)':>10} {'KL(p||D)':>10}")
    for k, p in cases.items():
        out(f"   {k:<44} {divergence.kl(D, p):10.3f} {divergence.kl(p, D):10.3f}")
    out("   (inf = the reverse direction is unbounded when p puts mass where D has none.)")
    out("")


@torch.no_grad()
def part_d(out, tok, n):
    from datasets import load_dataset
    out(f"D. The same loss on two kinds of text, {n} documents each; mean NLL per token in nats (bf16 autocast)")
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    paras = [t.strip() for t in wt["text"] if len(t.split()) > 60][:n]
    web = [T.Encoded(tok(p, add_special_tokens=False)["input_ids"][:256], [1.0] * min(256, len(tok(p, add_special_tokens=False)["input_ids"]))) for p in paras]
    alp = data.load_alpaca()
    _, val = data.split(alp, 0, 4000, seed=0)
    val = [e for e in val if len(e["response"]) < 600][:n]
    chat_all = pipeline.encode_examples(tok, val, mode="all", max_len=256)
    chat_asst = pipeline.encode_examples(tok, val, mode="assistant", max_len=256)
    out(f"   {'model':<22} {'web text':>10} {'chat, every token':>18} {'chat, assistant tokens':>24}")
    for label, repo in [("SmolLM2-135M (base)", lm.BASE), ("SmolLM2-135M-Instruct", lm.INSTRUCT)]:
        m = lm.load_model(repo).eval()
        r = [mean_nll(m, d, tok.pad_token_id)["token_nll"] for d in (web, chat_all, chat_asst)]
        out(f"   {label:<22} {r[0]:10.3f} {r[1]:18.3f} {r[2]:24.3f}")
        del m
    out("   Same function, same tokenizer, different text. The base model is better on web text; the instruct model is")
    out("   far better on assistant tokens. The instruct model also gives up some web-text loss. One model, one loss,")
    out("   and what it is good at is set by the distribution it was last trained on.")
    out("")


def part_e(out, tok, n_train, n_eval, seeds):
    sign = "Hope this helps."
    out(f"E. SFT on a mixture: {n_train} Alpaca examples, 80% of replies end with '{sign}' (never asked for), 20% do not.")
    out(f"   Evaluate on {n_eval} held-out prompts: share of replies ending with the sign-off. 3 epochs, lr 2e-5.")
    alp = [e for e in data.load_alpaca() if not e["context"] and 20 < len(e["response"]) < 500]
    rows = []
    for seed in seeds:
        tr, va = data.split(alp, n_train, n_eval, seed=seed)
        import random
        rng = random.Random(seed)
        tr = [{**e, "response": e["response"] + (" " + sign if rng.random() < 0.8 else "")} for e in tr]
        frac_in_data = sum(e["response"].endswith(sign) for e in tr) / len(tr)
        enc = pipeline.encode_examples(tok, tr, "assistant", 256)
        model, _ = pipeline.finetune(tok, enc, SFTConfig(lr=2e-5, epochs=3, seed=seed))
        res = {}
        for label, kw in [("greedy", {}), ("sampled T=1", {"sample": True, "seed": seed})]:
            g = pipeline.completions(model, tok, va, max_new=160, **kw)
            res[label] = sum(r["reply"].strip().endswith(sign) for r in g if r["stopped"]) / max(1, sum(r["stopped"] for r in g))
        rows.append((seed, frac_in_data, res["greedy"], res["sampled T=1"]))
        out(f"   seed {seed}: share in training data {frac_in_data:.3f}; share of replies with the sign-off, greedy {res['greedy']:.3f}, sampled {res['sampled T=1']:.3f}")
        del model
        torch.cuda.empty_cache()
    out(f"   mean over {len(seeds)} seeds: data {np.mean([r[1] for r in rows]):.3f}, greedy {np.mean([r[2] for r in rows]):.3f}, sampled {np.mean([r[3] for r in rows]):.3f}")
    out("   (Replies that hit the 160-token budget without ending are left out of the shares.)")
    out("   Greedy decoding picks the dominant mode (about 99% sign-off). Sampling keeps the minority style alive, and more")
    out("   of it than the data has: a quarter to two-fifths of sampled replies lack the sign-off, against 20% in the data.")
    out("   So the match with the mixture is rough, not calibrated: a 135M model sampled at T=1 also drifts for other reasons,")
    out("   and 3 epochs may not have fitted the 80/20 split. What the numbers support is the post's point in miniature:")
    out("   the loss keeps both modes, and nothing in it says which one is the unwanted one.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    out = _common.Report("02_same_loss")
    tok = lm.chat_tokenizer()
    part_a(out)
    part_b(out)
    part_c(out)
    part_d(out, tok, a.n_docs)
    rows = part_e(out, tok, a.n_train, a.n_eval, a.seeds)
    out.save()
    _common.save_json("02_same_loss", {"mixture_runs": rows})


if __name__ == "__main__":
    main()
