"""Section 01: without a nonlinearity, depth buys nothing. And not every nonlinearity is universal.

(a) Ten affine layers collapse into one affine map, exactly (up to float error).
(b) A deep linear net cannot separate two spirals. The same net with ReLU, tanh or GELU can.
(c) One hidden layer of x^3 - x is a cubic polynomial in its input at any width, so its error stops falling at
    the best cubic fit. ReLU and tanh keep improving with width. This is Leshno et al.'s non-polynomial condition.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from nonlin.data import two_spirals  # noqa: E402
from nonlin.mlp import DeepMLP  # noqa: E402
from nonlin.stats import Results  # noqa: E402


def collapse(layers: list[nn.Linear]) -> tuple[torch.Tensor, torch.Tensor]:
    """W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2), applied left to right."""
    w = torch.eye(layers[0].in_features, dtype=torch.float64)
    b = torch.zeros(layers[0].in_features, dtype=torch.float64)
    for layer in layers:
        lw, lb = layer.weight.double(), layer.bias.double()
        w, b = lw @ w, lw @ b + lb
    return w, b


def part_a(res: Results, seed: int) -> None:
    torch.manual_seed(seed)
    layers = [nn.Linear(16, 16) for _ in range(10)]
    x = torch.randn(256, 16, dtype=torch.float64)
    deep = x
    for layer in layers:
        deep = layer.double()(deep)
    w, b = collapse(layers)
    gap = (deep - (x @ w.T + b)).abs().max().item()
    res.print(f"(a) 10 affine layers vs their collapsed single map: max |difference| = {gap:.2e} (float64)")
    res.data["collapse_max_abs_diff"] = gap


def train_classifier(act: str, depth: int, steps: int, seed: int) -> tuple[float, float]:
    x, y = two_spirals(2000, seed=seed)
    xt, yt = two_spirals(1000, seed=seed + 100)
    torch.manual_seed(seed)
    init = "xavier" if act in ("identity", "tanh") else "he"
    model = DeepMLP(2, 64, 2, depth, act, init=init)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    with torch.no_grad():
        return ((model(x).argmax(1) == y).float().mean().item(), (model(xt).argmax(1) == yt).float().mean().item())


def part_b(res: Results, steps: int, seeds: list[int]) -> None:
    res.print(f"\n(b) Two spirals, 6 hidden layers of width 64, Adam lr 3e-3, {steps} full-batch steps, "
              f"seeds {seeds}")
    res.print(f"{'activation':<10} {'train acc':>10} {'test acc':>10}")
    out = {}
    for act in ["identity", "relu", "tanh", "gelu"]:
        accs = [train_classifier(act, 6, steps, s) for s in seeds]
        tr = sum(a for a, _ in accs) / len(accs)
        te = sum(b for _, b in accs) / len(accs)
        res.print(f"{act:<10} {tr:>10.3f} {te:>10.3f}")
        out[act] = {"train_acc": tr, "test_acc": te}
    res.data["spirals"] = out


def target_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(3 * math.pi * x) + 0.5 * x.abs()


def best_cubic_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    basis = torch.stack([x**k for k in range(4)], dim=1)
    coef = torch.linalg.lstsq(basis, y.unsqueeze(1)).solution
    return (basis @ coef).squeeze(1).sub(y).pow(2).mean().item()


def fit_1d(act: str, width: int, steps: int, seed: int, x: torch.Tensor, y: torch.Tensor) -> float:
    torch.manual_seed(seed)
    model = DeepMLP(1, width, 1, 1, act, init="xavier" if act != "relu" else "he")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(model(x.unsqueeze(1)).squeeze(1), y).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    with torch.no_grad():
        return nn.functional.mse_loss(model(x.unsqueeze(1)).squeeze(1), y).item()


def part_c(res: Results, steps: int, widths: list[int], seeds: list[int]) -> None:
    x = torch.linspace(-1, 1, 512, dtype=torch.float64).float()
    y = target_fn(x)
    floor = best_cubic_mse(x.double(), y.double())
    res.print(f"\n(c) One hidden layer, target sin(3 pi x) + 0.5|x| on [-1, 1], Adam, {steps} steps, "
              f"best of seeds {seeds}")
    res.print(f"Best possible cubic polynomial (least squares): MSE = {floor:.4f}")
    res.print(f"{'width':>6} " + " ".join(f"{a:>10}" for a in ["relu", "tanh", "cubic"]))
    out = {"best_cubic_mse": floor}
    for w in widths:
        row = {a: min(fit_1d(a, w, steps, s, x, y) for s in seeds) for a in ["relu", "tanh", "cubic"]}
        res.print(f"{w:>6} " + " ".join(f"{row[a]:>10.4f}" for a in ["relu", "tanh", "cubic"]))
        out[str(w)] = row
    res.data["approximation"] = out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    p.add_argument("--out", type=Path, default=LAB / "results")
    args = p.parse_args()
    steps = 50 if args.quick else args.steps
    seeds = list(range(1 if args.quick else args.seeds))
    widths = [4, 16] if args.quick else [4, 16, 64, 256]
    res = Results("01_linear_collapse", args.out)
    part_a(res, 0)
    part_b(res, steps, seeds)
    part_c(res, steps, widths, seeds)
    res.save()


if __name__ == "__main__":
    main()
