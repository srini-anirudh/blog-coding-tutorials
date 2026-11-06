"""Logits, softmax, entropy and temperature: the exact identities, then real entropy traces from SmolLM2-135M.

    python labs/choosing-the-next-token/scripts/01_entropy_temperature.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexttoken import dists, lm  # noqa: E402
from nexttoken.data import Report, prompt_and_reference  # noqa: E402

r = Report(__file__, "01_entropy_temperature")

r("== Exact identities on a 24-token Zipf-like logit vector (synthetic) ==")
z = -1.2 * torch.log(torch.arange(1, 25).float()) * 2
r(f"max |softmax(z) - softmax(z + 7.3)|      = {(dists.softmax(z) - dists.softmax(z + 7.3)).abs().max():.2e}")
for T in (0.5, 1.0, 2.0):
    fd = (dists.entropy(dists.softmax(z, T + 1e-3)) - dists.entropy(dists.softmax(z, T - 1e-3))) / 2e-3
    r(f"T={T}: dH/dT closed form {dists.dH_dT(z, T):.4f}   finite difference {fd:.4f}")
r1, rh = dists.odds_ratio(z, 1.0), dists.odds_ratio(z, 0.5)
r(f"odds ratio p2/p1 at T=1: {r1:.4f}   at T=0.5: {rh:.4f}   (T=1 value squared: {r1 ** 2:.4f})")
r(f"top-1 probability at T=0.1: {dists.softmax(z, 0.1)[0]:.6f}")
r("T      H (bits)  2^H")
for T in (0.05, 0.3, 0.7, 1.0, 1.5, 3.0, 6.0, 100.0):
    p = dists.softmax(z, T)
    r(f"{T:<6} {dists.entropy_bits(p):8.3f}  {dists.effective_support(p):6.2f}")
r("(uniform over 24 tokens would be 4.585 bits, 24 options)")

tok, model = lm.load(lm.BASE)
r("\n== Entropy along real text, SmolLM2-135M (next-token entropy after each token, bits) ==")
texts = [
    "import numpy as np\nimport matplotlib.pyplot as plt",
    "The most interesting thing about the model is that it",
    '{"name": "Alice", "age": 31}',
]
for text in texts:
    ids = tok(text, return_tensors="pt").input_ids
    p = torch.softmax(lm.all_logits(model, ids)[0], -1)
    h = dists.entropy_bits(p)
    r(f"\n{text!r}")
    for i in range(ids.shape[1]):
        nxt = tok.decode(ids[0, i + 1]) if i + 1 < ids.shape[1] else "(end)"
        p_next = f"{p[i, ids[0, i + 1]]:.3f}" if i + 1 < ids.shape[1] else "  -  "
        r(f"  after {tok.decode(ids[0, i])!r:<14} H={h[i]:6.2f} bits  2^H={2 ** h[i]:8.1f}  "
          f"p(actual next {nxt!r})={p_next}")

r("\n== Entropy over every position of 32 WikiText-2 passages (101 tokens each), SmolLM2-135M ==")
chunks = [torch.cat([p[0], ref]) for p, ref in prompt_and_reference(tok, 32, prompt_tokens=1, ref_tokens=100)]
hs = torch.cat([dists.entropy_bits(torch.softmax(lm.all_logits(model, x[None])[0], -1)) for x in chunks])
qs = torch.quantile(hs, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
r(f"positions {hs.numel()}  mean {hs.mean():.2f} bits  percentiles 10/25/50/75/90: "
  + " / ".join(f"{q:.2f}" for q in qs))
d = (hs[1:] - hs[:-1]).abs()
r(f"median |H_t+1 - H_t| between adjacent positions: {d.median():.2f} bits; fraction of adjacent pairs that differ "
  f"by more than 3 bits: {(d > 3).float().mean():.2f}")

r("\n== One temperature, two positions: the same change in T moves entropy by different amounts ==")


def logits_at(flat_index):
    c, i = divmod(flat_index, chunks[0].numel())
    return lm.all_logits(model, chunks[c][None])[0, i]


low, high = logits_at(int((hs - 0.5).abs().argmin())), logits_at(int((hs - 6).abs().argmin()))
r("T      H(position with H near 0.5)  H(position with H near 6)   dH/dT low   dH/dT high   (nats per unit T)")
for T in (0.3, 0.7, 1.0, 1.5, 3.0):
    hl = dists.entropy_bits(dists.softmax(low, T))
    hh = dists.entropy_bits(dists.softmax(high, T))
    r(f"{T:<6} {hl:12.3f} bits {hh:20.3f} bits {dists.dH_dT(low, T):14.4f} "
      f"{dists.dH_dT(high, T):12.4f}")
r.save()
