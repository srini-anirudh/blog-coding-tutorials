"""Meaning or pixels: what a pixel-trained, a caption-trained and a jointly trained encoder keep.

A toy version of the trade-off behind the Janus ablation and the post's Figure 5. Three encoders with the same
architecture are trained on the shapes images with different objectives (see seedraw/readers.py). Captions
name only the shape, not its position or brightness. Each frozen encoder is then scored on:

  reading:  a linear probe for the shape, fitted on 64 or 256 labelled images
  where:    a linear probe for the side (left or right), which captions never mention
  drawing:  test MSE of a fresh decoder trained on the frozen features
"""

from __future__ import annotations

import time

import _shared
from seedraw.common import Report, seed_all
from seedraw.readers import decoder_probe, features, linear_probe, train_encoder

STEPS = 2000
DECODER_STEPS = 1500


def main() -> None:
    seed_all(0)
    rep = Report("08_readers")
    train, test = _shared.splits()
    rep(f"Encoders: {STEPS} steps each. Decoder probe: {DECODER_STEPS} steps. Probes test on {len(test['images'])} images.")
    rep(f"{'encoder':10s} {'shape, 64 labels':>16s} {'shape, 256 labels':>17s} {'side, 256 labels':>16s} "
        f"{'decoder MSE':>11s} {'time':>5s}")
    out = {}
    for obj in ("pixel", "semantic", "both"):
        t0 = time.time()
        enc = train_encoder(obj, train["images"], train["shape"], STEPS)
        f_tr, f_te = features(enc, train["images"]), features(enc, test["images"])
        row = {
            "shape_probe_64": linear_probe(f_tr[:64], train["shape"][:64], f_te, test["shape"], 4),
            "shape_probe_256": linear_probe(f_tr[:256], train["shape"][:256], f_te, test["shape"], 4),
            "side_probe_256": linear_probe(f_tr[:256], train["side"][:256], f_te, test["side"], 2),
            "decoder_mse": decoder_probe(f_tr, train["images"], f_te, test["images"], DECODER_STEPS),
        }
        rep(f"{obj:10s} {row['shape_probe_64']:16.3f} {row['shape_probe_256']:17.3f} {row['side_probe_256']:16.3f} "
            f"{row['decoder_mse']:11.5f} {time.time() - t0:4.0f}s")
        out[obj] = row
    rep()
    rep("Chance: shape 0.25, side 0.50. A lower decoder MSE means the features keep more pixel detail.")
    rep.data = out
    rep.save()


if __name__ == "__main__":
    main()
