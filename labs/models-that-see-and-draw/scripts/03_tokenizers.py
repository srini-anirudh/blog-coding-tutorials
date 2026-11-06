"""Keeping the codebook alive: plain VQ against low-dim normalised VQ and FSQ.

Three tokenizers share one conv encoder and decoder (16 x 16 image -> 4 x 4 = 16 tokens, f = 4) and differ
only in the quantizer. All three have about 1,000 codes. The script trains each on the shapes dataset and
reports codebook usage, perplexity and reconstruction error on held-out images.
"""

from __future__ import annotations

import time

import torch

import _shared
from seedraw.common import Report, ascii_image, seed_all
from seedraw.tokenizers import codebook_stats, tokenize

KINDS = {
    "vq": "VQ, 1,024 codes of dim 64",
    "vq-lowdim": "VQ, 1,024 codes, dim 8, l2-normalised",
    "fsq": "FSQ, levels [8, 5, 5, 5] = 1,000 codes",
}


def main() -> None:
    seed_all(0)
    rep = Report("03_tokenizers")
    train, test = _shared.splits()
    rep(f"Shapes: {len(train['images'])} train, {len(test['images'])} test images, 16 x 16. "
        f"{_shared.TOKENIZER_STEPS} training steps each, batch 128.")
    rep()
    rep(f"{'tokenizer':40s} {'usage':>7s} {'perplexity':>10s} {'test MSE':>9s} {'time':>6s}")
    out = {}
    for kind, label in KINDS.items():
        seed_all(0)
        t0 = time.time()
        tok = _shared.get_tokenizer(kind, train)
        with torch.no_grad():
            idx = tokenize(tok, test["images"])
            mse = float(torch.nn.functional.mse_loss(tok.decode_indices(idx), test["images"]))
        stats = codebook_stats(idx, tok.quantizer.codebook)
        rep(f"{label:40s} {stats['usage']:7.1%} {stats['perplexity']:10.1f} {mse:9.5f} {time.time() - t0:5.0f}s")
        out[kind] = {"label": label, "codebook": tok.quantizer.codebook, "test_mse": mse, **stats}
        if kind == "fsq":
            with torch.no_grad():
                recon = tok.decode_indices(idx[:1]).clamp(0, 1)
            rep("  one test image, then its FSQ reconstruction from 16 integers " + str(idx[0].tolist()) + ":")
            rep(ascii_image(test["images"][0, 0]))
            rep(ascii_image(recon[0, 0]))
    rep()
    rep("Usage = share of codes chosen at least once on the test set. Perplexity = effective number of codes in use.")
    rep("Times include training only when no cached checkpoint existed.")
    rep.data = out
    rep.save()


if __name__ == "__main__":
    main()
