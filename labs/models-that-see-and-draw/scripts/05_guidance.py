"""Classifier-free guidance on image tokens.

Part 1 checks the two lines of the post's guidance box on twelve candidate tokens:
    l_g = l_u + s (l_c - l_u)
    log p_g(x) = log p(x) + s log p(c | x) + const
Part 2 samples the trained raster prior (trained with the condition dropped 10% of the time) at several
guidance scales, and measures prompt following, diversity and the cost in forward passes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import _shared
from seedraw.common import Report, seed_all
from seedraw.priors import cfg_logits, entropy_bits
from seedraw.readers import judge_accuracy

SCALES = [0.0, 1.0, 2.0, 3.0, 5.0]


def part1(rep: Report) -> dict:
    g = torch.Generator().manual_seed(3)
    l_u = torch.randn(12, generator=g)
    l_c = l_u + 1.2 * torch.randn(12, generator=g)  # illustrative logits, as in the post's figure
    rep("Part 1: twelve candidate codebook entries, illustrative logits.")
    same = torch.allclose(F.softmax(cfg_logits(l_u, l_c, 1.0), -1), F.softmax(l_c, -1))
    rep(f"  s = 1 gives back the conditional distribution: {same}")
    log_pu, log_pc = F.log_softmax(l_u, -1), F.log_softmax(l_c, -1)
    out = {"s1_equals_conditional": same, "entropy_bits": {}}
    for s in SCALES:
        guided = F.log_softmax(cfg_logits(l_u, l_c, s), -1)
        # Bayes: log p(c | x) = log p(x | c) - log p(x) + const, so the guided log-prob is log p(x) + s * that.
        bayes = log_pu + s * (log_pc - log_pu)
        gap = float((guided - F.log_softmax(bayes, -1)).abs().max())
        h = float(entropy_bits(cfg_logits(l_u, l_c, s)))
        out["entropy_bits"][s] = h
        rep(f"  s = {s:3.1f}: entropy {h:.2f} bits, top token p = {float(guided.exp().max()):.2f}, "
            f"max |guided - Bayes form| = {gap:.1e}")
    return out


def part2(rep: Report) -> list[dict]:
    train, _ = _shared.splits()
    tok = _shared.get_tokenizer("fsq", train)
    judge = _shared.get_judge(train)
    prior = _shared.get_prior("raster", tok, train)
    cond = torch.arange(4).repeat_interleave(128)
    rep()
    rep("Part 2: raster prior, 512 samples (128 per shape), temperature 1.")
    rep(f"  {'scale':>5s} {'shape acc':>9s} {'distinct':>8s} {'pixel spread':>12s} {'network evals':>13s}")
    rows = []
    for s in SCALES:
        seed_all(2)
        prior.passes = 0
        if s == 0.0:  # unconditional: sample with the null class
            toks = prior.sample_raster(torch.full_like(cond, prior.null_class))
        else:
            toks = prior.sample_raster(cond, scale=s)
        evals = prior.passes * (1 if s in (0.0, 1.0) else 2)  # guided passes run the batch twice
        imgs = tok.decode_indices(toks).clamp(0, 1)
        acc = judge_accuracy(judge, imgs, cond)["shape_acc"]
        distinct = len(torch.unique(toks, dim=0)) / len(toks)
        spread = float(torch.stack([imgs[cond == k].flatten(1).std(0).mean() for k in range(4)]).mean())
        rep(f"  {s:5.1f} {acc:9.3f} {distinct:8.1%} {spread:12.4f} {evals:13d}")
        rows.append({"scale": s, "shape_acc": acc, "distinct": distinct, "pixel_spread": spread, "network_evals": evals})
    rep("  scale 0 = unconditional samples. pixel spread = mean per-pixel std within each requested shape.")
    rep("  network evals counts a guided pass as two, one conditional and one unconditional.")
    return rows


def main() -> None:
    seed_all(0)
    rep = Report("05_guidance")
    p1 = part1(rep)
    p2 = part2(rep)
    rep.data = {"part1": p1, "part2": p2}
    rep.save()


if __name__ == "__main__":
    main()
