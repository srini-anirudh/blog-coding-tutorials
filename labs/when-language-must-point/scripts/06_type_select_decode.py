"""Type, select or decode (post Figure 6): three interfaces trained on the same toy referring task, scored on masks.

Type    location tokens for a box (cross-entropy); the box is then used as the region
Select  a class-agnostic proposer finds blobs; the model picks one region token (Groma style)
Decode  a <SEG> token's hidden state prompts a mask decoder, trained with BCE + Dice (LISA style)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointlab.report import Report  # noqa: E402
from pointlab.toy import box_masks, fill_ratios, make_dataset  # noqa: E402
from pointlab.train import (mask_metrics, predict_boxes, predict_decode, predict_select, train_decode,  # noqa: E402
                            train_select, train_typed)

ap = argparse.ArgumentParser()
ap.add_argument("--train", type=int, default=6000)
ap.add_argument("--test", type=int, default=1000)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--name", default="06_type_select_decode")
args = ap.parse_args()

R = Report(args.name)
t0 = time.time()
test = make_dataset(args.test, seed=999_999)
R(f"data: {args.train} training scenes, {args.test} test scenes; half the scenes let shapes touch or overlap")
R(f"each lane: {args.steps} steps of 64; seeds {args.seeds}; the test scenes are shared")

runs = {}
for seed in args.seeds:
    train = make_dataset(args.train, seed=1000 + seed)
    typed = train_typed("loc", train, args.steps, seed=seed, log=R)
    runs.setdefault("type (box from 4 loc tokens)", []).append(mask_metrics(box_masks(predict_boxes(typed, test)), test))
    select = train_select(train, args.steps, seed=seed, log=R)
    sel_masks, ceiling = predict_select(select, test)
    runs.setdefault("select (pick a proposal)", []).append(mask_metrics(sel_masks, test))
    decode = train_decode(train, args.steps, seed=seed, log=R)
    runs.setdefault("decode (<SEG> -> mask)", []).append(mask_metrics(predict_decode(decode, test), test))
runs["type, perfect box (oracle)"] = [mask_metrics(box_masks(test.box), test)]
res = {k: {c: sum(r[c] for r in v) / len(v) for c in v[0]} for k, v in runs.items()}
spread = {k: {c: (sum((r[c] - res[k][c]) ** 2 for r in v) / len(v)) ** 0.5 for c in v[0]} for k, v in runs.items()}

R("")
R(f"Test set ({args.test} scenes), mask IoU against the true region, mean over {len(args.seeds)} seeds")
cols = [("gIoU", "gIoU"), ("cIoU", "cIoU"), ("acc50", "Acc@0.5"), ("gIoU_compact", "gIoU compact"),
        ("gIoU_irregular", "gIoU irregular"), ("gIoU_separated", "gIoU separated"), ("gIoU_touching", "gIoU touching")]
R.table(["interface", *[c[1] for c in cols]], [[k, *[f"{v[c[0]]:.3f}" for c in cols]] for k, v in res.items()])

f = fill_ratios(test)
irr = [k in ("band", "bar", "wedge") for k in test.kind]
R("")
R(f"Mean fill ratio of the targets: compact {f[[not i for i in irr]].mean():.3f}, irregular {f[irr].mean():.3f}")
sep = ~test.touching
R(f"Select ceiling (best proposal has mask IoU >= 0.5): separated scenes {(ceiling[sep] >= 0.5).float().mean():.3f}, "
  f"touching scenes {(ceiling[~sep] >= 0.5).float().mean():.3f}")
R("Seed spread (std) of gIoU: " + ", ".join(f"{k} {spread[k]['gIoU']:.3f}" for k in res))
R(f"total time {time.time() - t0:.0f}s")
res["select_ceiling_acc50"] = {"separated": (ceiling[sep] >= 0.5).float().mean().item(),
                               "touching": (ceiling[~sep] >= 0.5).float().mean().item()}
R.save({"args": vars(args), "results": res, "std": spread, "runs": runs})
