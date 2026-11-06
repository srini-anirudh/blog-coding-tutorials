"""Softmax (CLIP) against sigmoid (SigLIP): gradient coupling, chunked computation, and a toy batch-size sweep.

Writes results/04_contrastive.txt and results/04_contrastive.json.
    --samples   training pairs seen per run (same for every batch size)
    --seeds     number of seeds per cell
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from vencoder.common import table, write_json, write_text  # noqa: E402
from vencoder.contrastive import ToyData, coupling, siglip_loss, siglip_loss_chunked, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 32, 128, 512])
    args = ap.parse_args()
    lines, out = [], {}

    c = coupling(n=8)
    out["coupling"] = c
    lines += ["Raise one off-diagonal logit of an 8 x 8 batch by 3. Other cells whose gradient changes:",
              f"  CLIP softmax: {c['clip']} of 63    SigLIP sigmoid: {c['siglip']} of 63", ""]

    g = torch.Generator().manual_seed(0)
    img = torch.nn.functional.normalize(torch.randn(256, 32, generator=g), dim=-1)
    txt = torch.nn.functional.normalize(torch.randn(256, 32, generator=g), dim=-1)
    t, b = torch.tensor(10.0), torch.tensor(-10.0)
    full = siglip_loss(img, txt, t, b)
    rows = []
    for k in (2, 4, 8):
        ch = siglip_loss_chunked(img, txt, t, b, k)
        rows.append([k, f"{full.item():.6f}", f"{ch.item():.6f}", f"{(full - ch).abs().item():.1e}",
                     f"{(256 // k) ** 2:,} of {256 * 256:,}"])
    out["chunked"] = rows
    lines += ["SigLIP loss on a 256 batch, full matrix against block by block:"]
    lines += table(rows, ["blocks per side", "full", "chunked", "|diff|", "logits held at once"]) + [""]

    data = ToyData()
    res = []
    t0 = time.time()
    for kind in ("clip", "siglip"):
        for batch in args.batches:
            for seed in range(args.seeds):
                res.append(train(kind, batch, args.samples, data, seed=seed))
                print(res[-1], flush=True)
    elapsed = time.time() - t0
    out["sweep"] = res
    rows = []
    for batch in args.batches:
        row = [batch, args.samples // batch]
        for kind in ("clip", "siglip"):
            accs = [r["top1"] for r in res if r["loss"] == kind and r["batch"] == batch]
            row.append(f"{statistics.mean(accs):.3f} +/- {statistics.pstdev(accs):.3f}")
        rows.append(row)
    lines += [f"Toy two-tower retrieval (top-1 among 2000 held-out pairs), {args.samples:,} pairs seen per run, "
              f"{args.seeds} seeds, mean +/- std. Total time {elapsed:.0f} s."]
    lines += table(rows, ["batch", "steps", "CLIP softmax", "SigLIP sigmoid"])

    print("\n".join(lines))
    write_text("04_contrastive.txt", lines)
    write_json("04_contrastive.json", out)


if __name__ == "__main__":
    main()
