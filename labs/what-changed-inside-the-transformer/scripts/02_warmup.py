"""Post-LN needs a warmup, Pre-LN does not.

Post: "Post-LN Transformers famously require a learning-rate warmup to train at all, and the deeper they get the
more fragile they become." Trains a 12-layer byte-level LM on Tiny Shakespeare with each norm placement, with and
without a linear warmup, at one learning rate, and prints the validation loss and loss spikes.

This is a small model (d=128, about 2.6M parameters) trained for a few hundred steps. It shows the mechanism, not
the size of the effect at the scale the post talks about.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from organs._paths import RESULTS
from organs.block import BlockConfig
from organs.report import Report
from organs.train import TrainConfig, load_text, train

RUNS = [("post", 0), ("post", 100), ("pre", 0), ("sandwich", 0), ("postbranch", 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="02_warmup")
    ap.add_argument("--runs", default=None, help="subset, e.g. post:0,post:100,pre:0 (placement:warmup)")
    args = ap.parse_args()
    rep = Report(args.out)
    text = load_text()
    rep(f"{args.layers} layers, d=128, 4 heads, RMSNorm, RoPE, SwiGLU. AdamW lr {args.lr}, batch 16 x {args.seq} bytes,")
    rep(f"{args.steps} steps, no clipping. Loss is bytes cross-entropy in nats (uniform guess = 5.55).")
    rep("")
    rows, curves = [], {}
    runs = [(r.split(":")[0], int(r.split(":")[1])) for r in args.runs.split(",")] if args.runs else RUNS
    for placement, warmup in runs:
        finals, spikes, worst, secs = [], [], [], 0.0
        for seed in range(args.seeds):
            mcfg = BlockConfig(layers=args.layers, placement=placement, max_seq=args.seq)
            tcfg = TrainConfig(steps=args.steps, seq=args.seq, lr=args.lr, warmup=warmup, seed=seed, eval_every=50)
            _, log = train(mcfg, tcfg, text)
            finals.append(log.final_val())
            spikes.append(log.spikes())
            worst.append(max(log.losses[20:] or log.losses))
            secs += log.seconds
            curves[f"{placement}-w{warmup}-s{seed}"] = log.val
        name = f"{placement}{' + warmup ' + str(warmup) if warmup else ''}"
        rows.append([name, " ".join(f"{v:.3f}" for v in finals), " ".join(str(s) for s in spikes),
                     " ".join(f"{w:.2f}" for w in worst), f"{secs / args.seeds:.0f}"])
        print(f"done {name}", flush=True)
    rep.table(["placement", "val loss per seed", "spikes per seed", "max train loss after step 20", "s per run"], rows)
    rep("")
    rep("A spike is a step whose loss is over 1.5x the median of the previous 20 steps.")
    (RESULTS / f"{args.out}_curves.json").write_text(json.dumps(curves))
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    torch.set_num_threads(min(8, torch.get_num_threads()))
    main()
