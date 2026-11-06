"""Grounding as next-token prediction on the toy referring task: digits vs location tokens, then an IoU reward.

Four models write the target's box on a 0-999 grid:
  digits     12 tokens from {0..9}, three per coordinate (text numbers, Qwen-VL style), cross-entropy
  loc        4 tokens from 1,000 location tokens (Florence-2 style), cross-entropy
  digits+RL  / loc+RL  the same models after GRPO-style fine-tuning with the box IoU as the reward
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointlab.report import Report  # noqa: E402
from pointlab.toy import make_dataset  # noqa: E402
from pointlab.train import box_metrics, coordinate_nll, predict_boxes, rl_finetune, train_typed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", type=int, default=6000)
ap.add_argument("--test", type=int, default=1000)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--rl-steps", type=int, default=400)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--name", default="05_train_coordinates")
args = ap.parse_args()

R = Report(args.name)
t0 = time.time()
test = make_dataset(args.test, seed=999_999)
R(f"data: {args.train} training scenes, {args.test} test scenes, 64 x 64 px, 3 shapes each, query = one colour")
R(f"CE: {args.steps} steps of 64; RL: {args.rl_steps} steps of 16 images x 8 samples; seeds {args.seeds}")
R("each seed draws its own training scenes and initialisation; the test scenes are shared")

runs = {}
for seed in args.seeds:
    train = make_dataset(args.train, seed=1000 + seed)
    for fmt in ("digits", "loc"):
        model = train_typed(fmt, train, args.steps, seed=seed, log=R)
        m = {**box_metrics(predict_boxes(model, test), test), "nll_per_coord": coordinate_nll(model, test)}
        runs.setdefault(fmt, []).append(m)
        rl = rl_finetune(model, train, args.rl_steps, seed=seed, log=R)
        m = {**box_metrics(predict_boxes(rl, test), test), "nll_per_coord": coordinate_nll(rl, test)}
        runs.setdefault(f"{fmt}+RL", []).append(m)
results = {k: {c: sum(r[c] for r in v) / len(v) for c in v[0]} for k, v in runs.items()}
spread = {k: {c: (sum((r[c] - results[k][c]) ** 2 for r in v) / len(v)) ** 0.5 for c in v[0]} for k, v in runs.items()}

R("")
R(f"Test set, greedy decoding ({args.test} scenes), mean over {len(args.seeds)} seeds (std in the JSON file)")
cols = [("box_iou", "mean box IoU"), ("acc50", "Acc@0.5"), ("acc75", "Acc@0.75"), ("wrong_object", "boxed a distractor"),
        ("far_miss_coords", "coords > 100 bins off"), ("median_err_bins", "median error (bins)"),
        ("nll_per_coord", "NLL per coordinate (nats)"), ("mask_iou_of_box_compact", "mask IoU of box: compact"),
        ("mask_iou_of_box_irregular", "mask IoU of box: irregular")]
R.table(["model", *[c[1] for c in cols]], [[k, *[f"{v[c[0]]:.3f}" for c in cols]] for k, v in results.items()])
R("")
R("Seed spread (std) of mean box IoU: " + ", ".join(f"{k} {spread[k]['box_iou']:.3f}" for k in results))
R(f"total time {time.time() - t0:.0f}s")
R.save({"args": vars(args), "results": results, "std": spread, "runs": runs})
