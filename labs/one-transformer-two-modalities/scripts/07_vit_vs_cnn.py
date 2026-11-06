"""Section 9, part 2, and section 4's learned table: a small ViT and a small CNN on CIFAR-10 subsets.

Same data, augmentation, optimizer and step budget for every run; only the training-set size and the model
change. The paper result (ViT overtakes ResNets only after JFT-300M pretraining) is far beyond this scale and is
not reproduced. What this shows: at small data the hard-coded prior wins, and a ViT's learned 1D position table
recovers the 2D grid on its own.

  python 07_vit_vs_cnn.py            the full run (GPU; 5,000 steps of batch 128 per model)
  python 07_vit_vs_cnn.py --device cpu --steps 1500 --batch 64 --sizes 2500 10000    a shorter run for CPU
  python 07_vit_vs_cnn.py --quick    a few steps on CPU, to check that everything runs

Needs CIFAR-10 (uoft-cs/cifar10) from the Hugging Face Hub.
"""

import argparse
import json

import _setup
import torch

from twomod import BlockConfig, SmallCNN, ViT
from twomod.front import count_params
from twomod.positions import alignment_score, interpolate_table, neighbour_score, similarity_by_distance
from twomod.train import TrainConfig, fit, load_cifar10

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
parser.add_argument("--out", default=None)
parser.add_argument("--device", default=None, help="cpu or cuda; default: cuda when available")
parser.add_argument("--steps", type=int, default=5_000)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--sizes", type=int, nargs="+", default=[2_500, 10_000, 50_000])
args = parser.parse_args()
device = args.device or ("cuda" if torch.cuda.is_available() and not args.quick else "cpu")
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
_setup.header(device)

train_x, train_y, test_x, test_y = (t.to(device) for t in load_cifar10())
if args.quick:
    sizes, cfg, test_x, test_y = [500], TrainConfig(steps=20, batch_size=32), test_x[:500], test_y[:500]
else:
    sizes, cfg = args.sizes, TrainConfig(steps=args.steps, batch_size=args.batch)
vit_cfg = BlockConfig(dim=128, n_heads=4)
print(f"device {device}; {cfg.steps} steps x batch {cfg.batch_size}, AdamW lr {cfg.lr}, wd {cfg.weight_decay}, "
      f"cosine with {cfg.warmup_frac:.0%} warmup; random crop + flip; test on {len(test_x):,} images")


def make(kind: str):
    if kind == "cnn":
        return SmallCNN()
    return ViT(vit_cfg, depth=6, image_size=32, patch=4, pos=kind.split("-")[1])


runs = [(n, kind) for n in sizes for kind in ("vit-learned", "cnn")]
runs += [(sizes[-1], "vit-sincos2d"), (sizes[-1], "vit-none")]
results = []
for n, kind in runs:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    subset = torch.randperm(len(train_x), generator=gen)[:n].to(device)  # the same subset for every model
    model = make(kind).to(device)
    row = {"model": kind, "params": count_params(model), **fit(model, train_x[subset], train_y[subset],
                                                               test_x, test_y, cfg)}
    if kind == "vit-learned":
        table = model.pos.detach().float().cpu()
        row["pos_neighbour_score"] = round(neighbour_score(table, 8, 8), 3)
        row["pos_cosine_by_distance_1_to_6"] = [round(v, 3) for v in similarity_by_distance(table, 8, 8)]
        row["pos_alignment_after_bicubic_to_16x16"] = round(
            alignment_score(table, interpolate_table(table, (8, 8), (16, 16)), (8, 8), (16, 16)), 3)
    results.append(row)
    print(json.dumps(row))

untrained = ViT(vit_cfg, depth=6).pos.detach()
baseline = round(neighbour_score(untrained, 8, 8), 3)
print(f"\nposition table neighbour score at initialisation (random): {baseline}")
print(f"\n{'train images':>12} {'model':<13} {'params':>9} {'train acc':>9} {'test acc':>8} {'seconds':>8}")
for r in results:
    print(f"{r['train_images']:>12,} {r['model']:<13} {r['params']:>9,} {r['train_acc']:>9.3f} "
          f"{r['test_acc']:>8.3f} {r['seconds']:>8.1f}")
if args.out:
    with open(args.out, "w") as f:
        json.dump({"hardware": _setup.hardware_note(device), "config": vars(cfg), "vit_block": vars(vit_cfg),
                   "pos_neighbour_score_at_init": baseline, "runs": results}, f, indent=1)
