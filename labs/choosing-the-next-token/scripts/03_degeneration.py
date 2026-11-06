"""The repetition attractor, then every sampler from the post on the same prompts, same model, same seeds.

    python labs/choosing-the-next-token/scripts/03_degeneration.py [--prompts 8] [--samples 4] [--tokens 80]

Generator: SmolLM2-135M. Prompts: the first 32 tokens of WikiText-2 test paragraphs. Metrics per decoder:
    rep-4    : fraction of 4-grams in a continuation that already appeared earlier in it (loops)
    distinct-2: unique bigrams / all bigrams, pooled over every continuation (diversity)
    tail     : fraction of emitted tokens to which the model itself gave p < 0.001 (draws from the tail)
    |C|      : mean candidate-set size at each step
    NLL      : mean per-token negative log-likelihood of the decoded text under Qwen2.5-0.5B, a model from a
               different family. Lower is not simply better: greedy text is very predictable. Compare to the human
               reference, the real WikiText continuation of the same prompts.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexttoken import lm, truncation as tr  # noqa: E402
from nexttoken.data import Report, prompt_and_reference  # noqa: E402
from nexttoken.metrics import distinct_n, repeated_ngram_fraction  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", type=int, default=8)
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--tokens", type=int, default=80)
args = ap.parse_args()

r = Report(__file__, "03_degeneration")
tok, model = lm.load(lm.BASE)

r("== The repetition attractor, SmolLM2-135M ==")
r("The same sentence is repeated n times after a short lead-in. Mean probability the model gives to the sentence's")
r("own tokens on the next repetition:")
sentences = ["The old bridge was closed after the storm damaged its northern supports.",
             "She packed the blue suitcase and walked slowly toward the station.",
             "Prices rose sharply in March, and the council delayed its decision again."]
r("n repeats so far   " + "   ".join(f"sentence {i + 1}" for i in range(len(sentences))))
for n in (0, 1, 2, 4, 8):
    row = []
    for s in sentences:
        ctx = tok("Notes from the meeting.\n" + (s + " ") * n, return_tensors="pt").input_ids
        cont = tok(s + " ", return_tensors="pt", add_special_tokens=False).input_ids
        row.append(lm.token_logprobs(model, ctx, cont).exp().mean().item())
    r(f"{n:<18} " + "   ".join(f"{v:10.3f}" for v in row))

DECODERS = [  # name, rule, param, temperature, order
    ("greedy", "full", None, 0.0, "truncate_then_temper"),
    ("pure sampling", "full", None, 1.0, "truncate_then_temper"),
    ("top-k 50", "top_k", 50, 1.0, "truncate_then_temper"),
    ("top-p 0.9", "top_p", 0.9, 1.0, "truncate_then_temper"),
    ("epsilon 3e-4", "epsilon", 3e-4, 1.0, "truncate_then_temper"),
    ("eta 3e-4", "eta", 3e-4, 1.0, "truncate_then_temper"),
    ("min-p 0.1", "min_p", 0.1, 1.0, "truncate_then_temper"),
    ("typical 0.9", "typical", 0.9, 1.0, "truncate_then_temper"),
    ("pure, T=1.5", "full", None, 1.5, "truncate_then_temper"),
    ("top-p 0.9, T=1.5, temper first", "top_p", 0.9, 1.5, "temper_then_truncate"),
    ("min-p 0.1, T=1.5, truncate first", "min_p", 0.1, 1.5, "truncate_then_temper"),
    ("min-p 0.1, T=1.5, temper first", "min_p", 0.1, 1.5, "temper_then_truncate"),
]

pairs = prompt_and_reference(tok, args.prompts, 32, args.tokens)
stok, scorer = lm.load(lm.SCORER)
r(f"\n== Samplers: {args.prompts} prompts x {args.samples} samples x {args.tokens} new tokens, seed 0 ==")
r(f"{'decoder':<34} {'rep-4':>6} {'dist-2':>7} {'tail':>6} {'|C|':>8} {'NLL':>6}")


def row(name, seqs, tails, sizes, nlls):
    rep = sum(repeated_ngram_fraction(s) for s in seqs) / len(seqs)
    tail = f"{sum(tails) / len(tails):6.3f}" if tails else "     -"
    size = f"{sum(sizes) / len(sizes):8.1f}" if sizes else "       -"
    r(f"{name:<34} {rep:6.3f} {distinct_n(seqs):7.3f} {tail} {size} {sum(nlls) / len(nlls):6.2f}")


ref_seqs = [ref.tolist() for _, ref in pairs]
row("human reference (WikiText)", ref_seqs, [], [],
    [lm.text_nll(stok, scorer, tok.decode(p[0]), tok.decode(ref)) for p, ref in pairs])

examples = {}
for name, rule, param, T, order in DECODERS:
    g = torch.Generator().manual_seed(0)
    seqs, tails, sizes, nlls = [], [], [], []
    n = 1 if T == 0 else args.samples
    for pi, (prompt, _) in enumerate(pairs):
        def pick(logits, _h):
            if T > 0:
                sizes.append(tr.candidate_mask(logits, rule, param, T, order).sum(-1).float().mean().item())
            return tr.sample(logits, rule, param, T, order, generator=g)

        gen, raw = lm.decode(model, prompt.expand(n, -1), pick, args.tokens, return_logits=True)
        p_emitted = torch.softmax(raw, -1).gather(-1, gen[..., None])[..., 0]
        tails.append((p_emitted < 1e-3).float().mean().item())
        for b in range(n):
            seqs.append(gen[b].tolist())
            nlls.append(lm.text_nll(stok, scorer, tok.decode(prompt[0]), tok.decode(gen[b])))
        if pi == 0:
            examples[name] = tok.decode(gen[0])
    row(name, seqs, tails, sizes, nlls)

r(f"\n== First continuation of prompt 1 under each decoder ==\nprompt: {tok.decode(pairs[0][0][0])!r}")
for name, text in examples.items():
    r(f"\n[{name}]\n{text[:400]!r}")
r.save()
