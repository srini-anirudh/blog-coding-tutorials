"""Section "Learning vision from language": CLIP's softmax loss, SigLIP's sigmoid loss, and the modality gap.

    python labs/teaching-language-models-to-see/scripts/s1_contrastive.py [--quick]

1. The sigmoid loss splits into blocks with no error; the softmax loss does not.
2. Two random towers already put images and captions in two narrow cones, apart from each other
   (Liang et al.'s cone effect), and we track that gap through contrastive training.
3. A toy batch-size sweep for both losses at a fixed number of examples seen. Toy data only.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seelab.common import Report  # noqa: E402
from seelab.contrastive import (  # noqa: E402
    ToyPairs, TwoTower, clip_loss, clip_loss_chunked_wrong, modality_gap, retrieval_top1, siglip_loss,
    siglip_loss_chunked, train_toy,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer seeds and examples (for a smoke test)")
    args = ap.parse_args()
    torch.manual_seed(0)
    rep = Report("s1_contrastive")
    out: dict = {}

    rep("## 1. Does the loss split into blocks?  B = 96, d = 32, blocks of 32 (a 3 x 3 grid of blocks)")
    u, v = F.normalize(torch.randn(96, 32, dtype=torch.float64), dim=-1), F.normalize(torch.randn(96, 32, dtype=torch.float64), dim=-1)
    v = F.normalize(v + 2 * u, dim=-1)  # make the diagonal the right answer, as after some training
    full, chunked = siglip_loss(u, v, 10.0, -10.0), siglip_loss_chunked(u, v, 10.0, -10.0, chunk=32)
    c_full, c_wrong = clip_loss(u, v, 0.07), clip_loss_chunked_wrong(u, v, 0.07, chunk=32)
    rep(f"sigmoid  full {full.item():.10f}   blockwise {chunked.item():.10f}   |diff| {abs(full - chunked).item():.1e}")
    rep(f"softmax  full {c_full.item():.10f}   per-block softmax {c_wrong.item():.10f}   |diff| {abs(c_full - c_wrong).item():.1e}")
    rep("The softmax needs every caption in a row to normalise; a block that sees 32 of 96 captions gets an easier task.")
    out["split"] = {"sigmoid_full": full.item(), "sigmoid_blockwise": chunked.item(),
                    "softmax_full": c_full.item(), "softmax_per_block": c_wrong.item()}

    rep("\n## 2. The modality gap (cone effect).  Toy paired data, held-out 1,000 pairs, 4-layer ReLU towers")
    data = ToyPairs()
    g = torch.Generator().manual_seed(123)
    img_te, txt_te = data.sample(1000, g)
    rep(f"{'model':<34} {'gap':>6} {'cone img':>9} {'cone txt':>9} {'cos pair':>9} {'cos all':>8} {'top-1':>6}")
    gap_rows = {}
    for name in ("random init, seed 0", "random init, seed 1"):
        torch.manual_seed(int(name[-1]))
        m = TwoTower(64, 32)
        with torch.no_grad():
            r = modality_gap(*m.embed(img_te, txt_te))
        r["top1"] = retrieval_top1(m, img_te, txt_te)
        gap_rows[name] = r
    with torch.no_grad():  # random directions, for scale: no cone, no gap
        r = modality_gap(torch.randn(1000, 32), torch.randn(1000, 32))
    r["top1"] = float("nan")
    gap_rows["random unit vectors (reference)"] = r
    steps = 600 if args.quick else 3000
    for loss in ("clip", "siglip"):
        m, _ = train_toy(loss, 256, steps, data)
        with torch.no_grad():
            r = modality_gap(*m.embed(img_te, txt_te))
        r["top1"] = retrieval_top1(m, img_te, txt_te)
        gap_rows[f"{loss} trained, B=256, {steps} steps"] = r
    for name, r in gap_rows.items():
        rep(f"{name:<34} {r['gap']:6.3f} {r['cone_image']:9.3f} {r['cone_text']:9.3f} {r['cos_matched']:9.3f} "
            f"{r['cos_image_text_all']:8.3f} {r['top1']:6.3f}")
    rep("gap = distance between the two centroids on the unit sphere. cone = mean cosine between different vectors")
    rep("of one modality. cos pair = matched image-caption cosine; cos all = mean over every image-caption pair.")
    out["gap"] = gap_rows

    rep("\n## 3. Toy batch-size sweep at a fixed number of examples seen. Top-1 retrieval among 1,000 held-out pairs")
    seen = 32_000 if args.quick else 128_000
    seeds = (0,) if args.quick else (0, 1, 2)
    rep(f"{seen:,} training examples per run, {len(seeds)} seeds, AdamW lr 1e-3, mean (min-max)")
    sweep = {}
    start = time.time()
    for bs in (8, 32, 128, 512):
        row = {}
        for loss in ("clip", "siglip"):
            accs = [retrieval_top1(train_toy(loss, bs, seen // bs, data, seed=s)[0], img_te, txt_te) for s in seeds]
            row[loss] = accs
        sweep[bs] = row
        fmt = lambda a: f"{statistics.mean(a):.3f} ({min(a):.3f}-{max(a):.3f})"  # noqa: E731
        rep(f"B = {bs:<4}  softmax {fmt(row['clip'])}   sigmoid {fmt(row['siglip'])}")
    rep(f"sweep time {time.time() - start:.0f} s. This is a 16-dimensional toy, not a test of the paper's claim at scale.")
    out["batch_sweep"] = {"examples_seen": seen, "seeds": list(seeds), "results": sweep}
    rep.save(out)


if __name__ == "__main__":
    main()
