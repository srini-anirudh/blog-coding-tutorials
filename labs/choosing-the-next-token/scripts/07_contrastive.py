"""Contrastive decoding (expert SmolLM2-360M minus amateur SmolLM2-135M) and contrastive search, side by side.

    python labs/choosing-the-next-token/scripts/07_contrastive.py [--prompts 8] [--tokens 100]

Metrics as in 03_degeneration.py: rep-4, distinct-2, and NLL under Qwen2.5-0.5B. Small models, small sample: this
shows the mechanism, not the gains reported in the papers.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexttoken import contrastive, lm, truncation as tr  # noqa: E402
from nexttoken.data import Report, prompt_and_reference  # noqa: E402
from nexttoken.metrics import distinct_n, repeated_ngram_fraction  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", type=int, default=8)
ap.add_argument("--tokens", type=int, default=100)
args = ap.parse_args()

r = Report(__file__, "07_contrastive")
tok, amateur = lm.load(lm.BASE)
_, expert = lm.load(lm.LARGE)
stok, scorer = lm.load(lm.SCORER)
pairs = prompt_and_reference(tok, args.prompts, 32, args.tokens)

r("== Why the plausibility constraint is needed: next token after one open-ended prompt ==")
prompt = tok("My favourite thing about the old town is the", return_tensors="pt").input_ids
le = torch.log_softmax(lm.next_logits(expert, prompt), -1)[0]
la = torch.log_softmax(lm.next_logits(amateur, prompt), -1)[0]
for label, alpha in [("no constraint", None), ("alpha = 0.1", 0.1)]:
    s = contrastive.contrastive_scores(le[None], la[None], alpha)[0]
    top = [i for i in torch.topk(s, 5).indices if torch.isfinite(s[i])]
    r(f"{label:<14} top tokens by log p_e - log p_a: "
      + ", ".join(f"{tok.decode(i)!r} (p_e {le[i].exp():.1e}, p_a {la[i].exp():.1e})" for i in top))

g = torch.Generator().manual_seed(0)
DECODERS = {
    "greedy, 135M": lambda p: lm.decode(amateur, p, lambda z, h: z.argmax(-1), args.tokens),
    "greedy, 360M": lambda p: lm.decode(expert, p, lambda z, h: z.argmax(-1), args.tokens),
    "top-p 0.95, 360M": lambda p: lm.decode(expert, p, lambda z, h: tr.sample(z, "top_p", 0.95, generator=g),
                                            args.tokens),
    "contrastive decoding, alpha 0.1": lambda p: contrastive.contrastive_decode(expert, amateur, p, args.tokens, 0.1),
    "contrastive decoding, no alpha": lambda p: contrastive.contrastive_decode(expert, amateur, p, args.tokens, None),
    "contrastive search, 360M k=5 a=0.6": lambda p: contrastive.contrastive_search(expert, p, args.tokens, 5, 0.6),
}
r(f"\n== {args.prompts} WikiText-2 prompts x {args.tokens} new tokens ==")
r(f"{'decoder':<36} {'rep-4':>6} {'dist-2':>7} {'NLL':>6}")
ref_nll = [lm.text_nll(stok, scorer, tok.decode(p[0]), tok.decode(ref)) for p, ref in pairs]
r(f"{'human reference (WikiText)':<36} {sum(repeated_ngram_fraction(x.tolist()) for _, x in pairs) / len(pairs):6.3f} "
  f"{distinct_n([x.tolist() for _, x in pairs]):7.3f} {sum(ref_nll) / len(ref_nll):6.2f}")
examples = {}
for name, run in DECODERS.items():
    seqs, nlls = [], []
    for prompt, _ in pairs:
        gen = run(prompt)[0].tolist()
        seqs.append(gen)
        nlls.append(lm.text_nll(stok, scorer, tok.decode(prompt[0]), tok.decode(gen)))
    examples[name] = tok.decode(seqs[0])
    r(f"{name:<36} {sum(map(repeated_ngram_fraction, seqs)) / len(seqs):6.3f} {distinct_n(seqs):7.3f} "
      f"{sum(nlls) / len(nlls):6.2f}")
r(f"\n== Continuations of prompt 1 ==\nprompt: {tok.decode(pairs[0][0][0])!r}")
for name, text in examples.items():
    r(f"\n[{name}]\n{text[:400]!r}")
r.save()
