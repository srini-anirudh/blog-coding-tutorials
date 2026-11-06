"""The alpha/r question (post: "Scaling, initialization, and the alpha/r question").

A frozen 256 x 256 layer must learn a full-rank change with a slowly decaying spectrum, so more rank genuinely helps.
Every run uses the same alpha (16), the same Adam learning rate and the same number of steps. Only the rank and the
scaling rule change. The best a rank-r adapter can do here is known exactly: the Eckart-Young tail of the change.

Under alpha/r the effective step size of the adapter shrinks as r grows, so high ranks learn slowly.
Under alpha/sqrt(r) (rsLoRA) it does not, and higher rank keeps paying off. The last part tries LoRA+'s learning-rate
ratio for B over A.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loralab.lora import LoRALinear
from loralab.report import Report, table
from loralab.spectrum import eckart_young_error, matrix_with_spectrum

D = 256
CHECKPOINTS = (50, 200)


def task(seed=0):
    g = torch.Generator().manual_seed(seed)
    base = nn.Linear(D, D, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.randn(D, D, generator=g) / D ** 0.5)
    sig = 2.0 * torch.arange(1, D + 1, dtype=torch.float64) ** -0.5   # slow power-law decay, full rank
    delta = matrix_with_spectrum(sig, D, D, seed=seed + 100).float()
    return base, delta


def train(base, delta, r, scaling, steps, lr=1e-3, lr_ratio=1.0, alpha=16.0, seed=0, batch=256):
    torch.manual_seed(seed)
    layer = LoRALinear(nn.Linear(D, D, bias=False), r=r, alpha=alpha, scaling=scaling)
    layer.base.weight.data.copy_(base.weight)
    opt = torch.optim.Adam([{"params": [layer.lora_A], "lr": lr}, {"params": [layer.lora_B], "lr": lr * lr_ratio}])
    g = torch.Generator().manual_seed(seed + 1)
    curve = {}
    for step in range(1, steps + 1):
        x = torch.randn(batch, D, generator=g)
        y = x @ (base.weight + delta).T
        loss = (layer(x) - y).pow(2).sum(1).mean()   # E ||(Delta - s B A) x||^2 = ||Delta - s B A||_F^2 for x ~ N(0, I)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in CHECKPOINTS or step == steps:
            curve[step] = (delta - layer.delta_weight()).norm().item()
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    rep = Report("04_scaling")
    base, delta = task()
    dnorm = delta.norm().item()
    rep(f"target change: 256 x 256, full rank, sigma_i = 2 i^-0.5, ||Delta||_F = {dnorm:.3f}")
    rep(f"Adam lr 1e-3, alpha 16, {args.steps} steps, batch 256, {args.seeds} seeds; error is ||Delta - s B A||_F")
    ranks = (2, 8, 32, 128, 256)
    rows, data = [], {}
    for scaling in ("standard", "rslora"):
        for r in ranks:
            runs = [train(base, delta, r, scaling, args.steps, seed=s) for s in range(args.seeds)]
            mean = {k: sum(c[k] for c in runs) / len(runs) for k in runs[0]}
            spread = max(c[200] for c in runs) - min(c[200] for c in runs)
            floor = eckart_young_error(delta.double(), r)
            s = 16 / r if scaling == "standard" else 16 / r ** 0.5
            got = (dnorm - mean[200]) / (dnorm - floor)   # share of the reachable improvement done by step 200
            rows.append([scaling, r, f"{s:.3g}", mean[50], mean[200], spread, f"{got:.0%}", mean[args.steps], floor])
            data[f"{scaling}_r{r}"] = {"scale": s, "error": mean, "seed_spread_at_200": spread,
                                       "share_of_reachable_at_200": got, "eckart_young_floor": floor}
    for line in table(rows, ["scaling", "rank", "s", "error @ 50", "error @ 200", "spread @ 200",
                             "reachable done @ 200", f"error @ {args.steps}", "best possible (E-Y)"]):
        rep(line)
    rep("error is the mean over seeds; spread is max - min over seeds; 'reachable done' is "
        "(||Delta|| - error) / (||Delta|| - E-Y floor)")
    rep.data["sweep"] = data

    rep("\nLoRA+ (lr_B = ratio x lr_A), rank 32, standard scaling, seed 0")
    rows = []
    for ratio in (1, 4, 16):
        c = train(base, delta, 32, "standard", args.steps, lr_ratio=ratio)
        rows.append([ratio, c[50], c[200], c[args.steps]])
        data[f"loraplus_ratio{ratio}"] = c
    for line in table(rows, ["lr_B / lr_A", "error @ 50", "error @ 200", f"error @ {args.steps}"]):
        rep(line)
    rep.save()


if __name__ == "__main__":
    main()
