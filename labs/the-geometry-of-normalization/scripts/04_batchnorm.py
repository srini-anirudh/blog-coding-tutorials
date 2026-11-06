"""Section 04: batch statistics are a noisy estimator, and padding contaminates them.

Part A (Fig 04). Draw mini-batches from a fixed population, N(2, 3^2) per channel. For each batch size, measure
  - the spread of the batch mean (theory: 3 / sqrt(B)),
  - the gap between BatchNorm's training-mode output (batch stats) and eval-mode output (running EMA stats).
Part B. Variable-length sequences padded with zeros. BatchNorm over (sample, position) counts the pads.
"""

import _setup  # noqa: F401
import torch
from torch import nn

from normgeo.report import Report

torch.manual_seed(0)
MU, SIGMA, C = 2.0, 3.0, 16

with Report("04_batchnorm") as say:
    say(f"Part A. Population N({MU}, {SIGMA}^2) per channel, {C} channels, 2000 batches per size.")
    say(f"{'batch':>6}{'std of batch mean':>19}{'3/sqrt(B)':>11}{'mean |batch std - 3|':>22}{'|train - eval| output':>23}{'EMA |mean err|, |std err|':>28}")
    rows = []
    for B in [2, 4, 8, 32, 128, 512]:
        draws = torch.randn(2000, B, C) * SIGMA + MU
        mean_spread = draws.mean(1).std().item()
        std_err = (draws.std(1, unbiased=False) - SIGMA).abs().mean().item()
        bn = nn.BatchNorm1d(C, affine=False, momentum=0.1)
        bn.train()
        for batch in draws[:1000]:
            bn(batch)  # fills the running statistics
        gaps = []
        for batch in draws[1000:]:
            bn.train()
            y_train = bn(batch)
            bn.eval()
            gaps.append((y_train - bn(batch)).abs().mean().item())
        gap = sum(gaps) / len(gaps)
        ema_mu = (bn.running_mean - MU).abs().mean().item()
        ema_sd = (bn.running_var.sqrt() - SIGMA).abs().mean().item()
        rows.append({"batch": B, "mean_spread": mean_spread, "std_err": std_err, "train_eval_gap": gap, "ema_mean_err": ema_mu, "ema_std_err": ema_sd})
        say(f"{B:>6}{mean_spread:>19.3f}{SIGMA / B**0.5:>11.3f}{std_err:>22.3f}{gap:>23.3f}{ema_mu:>16.3f}, {ema_sd:.3f}")
    say("The training-time function is a different function from the eval-time one. The gap grows as the batch shrinks.")
    say("The EMA (momentum 0.1) averages about 19 batches, so its error is smaller than one batch's, but at batch 2")
    say("it is still far from the truth: the eval-time function is itself noisy at small batch.")

    say()
    say("Part B. 8 sequences, lengths 4 to 64, padded to 64 with zeros. True per-feature mean 1.0, std 1.0.")
    lengths = torch.tensor([4, 8, 12, 16, 24, 32, 48, 64])
    T = int(lengths.max())
    x = torch.randn(8, T, C) + 1.0
    mask = torch.arange(T)[None, :] < lengths[:, None]  # (8, T), True on real tokens
    x = x * mask[..., None]
    with_pads = x.mean((0, 1))
    masked = x.sum((0, 1)) / mask.sum()
    say(f"  BatchNorm mean with pads counted: {with_pads.mean():.3f}. With pads masked out: {masked.mean():.3f}.")
    say(f"  Fraction of the (sample, position) grid that is padding: {1 - mask.float().mean():.2f}.")
    per_pos = mask.sum(0)
    say("  Real sequences that contribute at each position: "
        + ", ".join(f"pos {p}: {int(per_pos[p])}" for p in [0, 8, 16, 32, 48, 63]))
    say("  With masking, late positions get statistics from 1 to 2 sequences. LayerNorm is per token: pads never enter.")
    say.json({"part_a": rows, "pad_mean": with_pads.mean().item(), "masked_mean": masked.mean().item()})
