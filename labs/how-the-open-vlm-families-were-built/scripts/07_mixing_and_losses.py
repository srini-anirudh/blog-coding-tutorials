"""Mixing and loss tips that recur across the families, run small.

Offline. Source sizes are the LLaVA-1.5 SFT mix as the post lists it (VQAv2 83K, GQA 72K, OCR-VQA 80K,
RefCOCO 48K, Visual Genome 86K, ShareGPT 40K; the rest of the 665K is grouped as "other"). Response lengths in
part 2 are the post's own example (a one-word answer against a 200-word caption), not measured.
"""

import _path  # noqa: F401

import torch

from vlmfam import mixing as M
from vlmfam.report import Report

torch.manual_seed(0)
r = Report("07_mixing_and_losses")
SIZES = {"VQAv2": 83_000, "GQA": 72_000, "OCR-VQA": 80_000, "RefCOCO": 48_000, "VG": 86_000, "ShareGPT": 40_000}
SIZES["other"] = 665_000 - sum(SIZES.values())

# ---------------------------------------------------------------- part 1: caps and square-root sampling
r("Part 1. Sampling shares for the LLaVA-1.5 665K mix")
plans = {"proportional": M.capped_weights(SIZES), "sqrt (Molmo)": M.capped_weights(SIZES, strategy="sqrt"),
         "cap 60K (Cambrian-style)": M.capped_weights(SIZES, cap=60_000),
         "cap 60K + sqrt": M.capped_weights(SIZES, cap=60_000, strategy="sqrt")}
rows = [[k, f"{n // 1000}K"] + [f"{p[k]:.1%}" for p in plans.values()] for k, n in SIZES.items()]
r.table(["source", "samples"] + list(plans), rows)
r("  Square-root sampling and caps both move weight from the largest source to the smallest ones.")

# ---------------------------------------------------------------- part 2: samples are not tokens
r("")
r("Part 2. A 50:50 mixture in samples, seen as tokens (post: 'Mixtures in tokens')")
shares = M.token_shares({"short VQA answers": 0.5, "long captions": 0.5},
                        {"short VQA answers": 1, "long captions": 200})
for k, v in shares.items():
    r(f"  {k:<18} 50.0% of samples  ->  {v:.1%} of supervised tokens")

# ---------------------------------------------------------------- part 3: loss normalisation
r("")
r("Part 3. One batch with responses of 1, 10 and 200 tokens. Share of the loss gradient each sample gets")
lengths = [1, 10, 200]
vocab, L = 50, max(lengths)
labels = torch.full((3, L), -100)
for i, n in enumerate(lengths):
    labels[i, :n] = torch.randint(0, vocab, (n,))
base = torch.randn(3, L, vocab)
rows = []
for mode in ("token", "sample", "sqrt"):
    logits = base.clone().requires_grad_(True)
    M.sft_loss(logits, labels, mode).backward()
    # per-sample gradient mass: sum of |dL/dlogit| over that sample's tokens, as a share of the total
    mass = logits.grad.abs().sum(dim=(1, 2))
    measured = (mass / mass.sum()).tolist()
    closed = M.sample_weights(lengths, mode)
    rows.append([mode] + [f"{m:.1%} ({c:.1%})" for m, c in zip(measured, closed)])
r.table(["normalisation", "1 token", "10 tokens", "200 tokens"], rows)
r("  Measured from autograd; in brackets the closed form L, 1 or sqrt(L) normalised. They differ slightly because")
r("  the gradient size of each token depends on its own logits. Per-token loss lets the long caption take almost")
r("  all of the gradient; per-sample lets one-word answers weigh as much as captions; sqrt sits between.")

# ---------------------------------------------------------------- part 4: modality warm-up
r("")
r("Part 4. DeepSeek-VL modality warm-up to a 70% text share (warm-up length is illustrative: 1,000 steps)")
r("  step:       " + "  ".join(f"{s:>5}" for s in (0, 250, 500, 750, 1000, 2000)))
r("  text share: " + "  ".join(f"{M.modality_warmup(s, 1000, 0.7):>5.0%}" for s in (0, 250, 500, 750, 1000, 2000)))
r.save({"sampling": plans, "token_shares": shares, "loss_rows": rows})
