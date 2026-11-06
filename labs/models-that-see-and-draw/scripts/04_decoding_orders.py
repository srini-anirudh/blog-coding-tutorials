"""Two orders for producing the same token image: raster autoregression and masked parallel decoding.

Both priors model the 16 FSQ tokens of a shape image, conditioned on the shape class. The raster prior is
causal and needs one forward pass per token. The MaskGIT prior is bidirectional and fills a growing share of
the grid per pass. A separate judge CNN grades whether each decoded image shows the requested shape.

The script also prints the pass counts of the post's Figure 2 for a 16 x 16 token grid. Those are arithmetic.
"""

from __future__ import annotations

import time

import torch

import _shared
from seedraw.common import Report, ascii_image, seed_all
from seedraw.priors import pass_counts
from seedraw.readers import judge_accuracy

N_PER_CLASS = 128


def main() -> None:
    seed_all(0)
    rep = Report("04_decoding_orders")
    train, test = _shared.splits()
    tok = _shared.get_tokenizer("fsq", train)
    judge = _shared.get_judge(train)
    rep(f"Judge accuracy on real test images: {judge_accuracy(judge, test['images'], test['shape'], test['side'])}")
    with torch.no_grad():
        recon = tok.decode_indices(_shared.tokenize(tok, test["images"])).clamp(0, 1)
    ceiling = judge_accuracy(judge, recon, test["shape"])["shape_acc"]
    rep(f"Judge shape accuracy on FSQ reconstructions of test images (the tokenizer's ceiling): {ceiling:.3f}")
    rep()

    rep("Pass counts for a 16 x 16 token grid (arithmetic, as in the post's Figure 2):")
    counts = pass_counts(16, 12, [1, 2, 3, 4, 6, 8, 10, 13, 16], 50)
    for k, v in counts.items():
        rep(f"  {k:52s} {v:4d} passes")
    rep()

    cond = torch.arange(4).repeat_interleave(N_PER_CLASS)
    rows = []
    rep(f"Sampling {len(cond)} images ({N_PER_CLASS} per shape), no guidance, temperature 1.")
    rep(f"{'order':32s} {'passes':>6s} {'time (s)':>8s} {'shape acc':>9s} {'distinct':>8s}")
    for order, steps_list in [("raster", [None]), ("maskgit", [1, 2, 4, 8])]:
        prior = _shared.get_prior(order, tok, train)
        for steps in steps_list:
            seed_all(1)
            prior.passes = 0
            t0 = time.time()
            if order == "raster":
                toks = prior.sample_raster(cond)
                name = "raster autoregression"
            else:
                toks = prior.sample_maskgit(cond, steps=steps)
                name = f"MaskGIT, {steps} rounds"
            dt = time.time() - t0
            imgs = tok.decode_indices(toks).clamp(0, 1)
            acc = judge_accuracy(judge, imgs, cond)["shape_acc"]
            distinct = len(torch.unique(toks, dim=0)) / len(toks)
            rep(f"{name:32s} {prior.passes:6d} {dt:8.2f} {acc:9.3f} {distinct:8.1%}")
            rows.append({"order": name, "passes": prior.passes, "seconds": dt, "shape_acc": acc, "distinct": distinct})
            if order == "raster" or steps == 4:
                rep("  first sample (requested: square):")
                rep(ascii_image(imgs[0, 0]))
    rep()
    rep("passes = sequential forward passes of the network for the whole batch. shape acc = share of samples the judge "
        "reads as the requested shape. distinct = share of unique token grids.")
    rep.data = {"judge_on_reconstructions": ceiling, "figure2_pass_counts": counts, "rows": rows}
    rep.save()


if __name__ == "__main__":
    main()
