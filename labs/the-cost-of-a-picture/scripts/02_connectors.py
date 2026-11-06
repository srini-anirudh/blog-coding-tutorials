"""Five connectors on a toy patch grid: how many tokens each emits, and which task each can still solve."""

import argparse

import torch
from _common import write

from pixelcost import connectors as cn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    x = torch.randn(2, 64, 8)
    folded = cn.pixel_shuffle(x, 2)
    pooled = cn.avg_pool(x, 2)
    lines = [f"pixel shuffle 2: {tuple(x.shape)} -> {tuple(folded.shape)}; unshuffle recovers the input exactly: "
             f"{torch.equal(cn.unshuffle(folded, 2), x)}",
             f"average pool 2: {tuple(x.shape)} -> {tuple(pooled.shape)}; numbers kept: {pooled.numel()} of {x.numel()}",
             "",
             f"Toy task accuracy (test set of 512, chance 0.25), {args.steps} steps, mean over {args.seeds} seeds",
             f"{'connector':<11} {'tokens':>6} {'glyph (detail inside a 2x2 block)':>34} {'blob (which quadrant)':>22}"]
    rows = []
    for kind in ["mlp", "shuffle", "pool", "resampler", "conv"]:
        res = {}
        for task in ["glyph", "blob"]:
            runs = [cn.train_and_eval(kind, task, args.steps, seed) for seed in range(args.seeds)]
            res[task] = sum(r["test_accuracy"] for r in runs) / len(runs)
            res["tokens"] = runs[0]["tokens"]
            res[task + "_runs"] = [r["test_accuracy"] for r in runs]
        rows.append({"connector": kind, **res})
        lines.append(f"{kind:<11} {res['tokens']:>6} {res['glyph']:>34.3f} {res['blob']:>22.3f}")
    lines.append("\nToy model: 8 x 8 grid of 8-dim patches, connector, then a one-layer transformer reader. "
                 "Not a VLM measurement.")
    write("02_connectors", "\n".join(lines), {"rows": rows})


if __name__ == "__main__":
    main()
