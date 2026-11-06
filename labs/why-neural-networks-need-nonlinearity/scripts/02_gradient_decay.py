"""Sections 02 and 08: watching a gradient die, and why the initialisation must match the activation.

(a) One backward pass through an 18-layer MLP. Per-layer weight-gradient norms, and the fitted decay in decades
    per layer. The post's Fig 02 says sigmoid with Xavier loses about 0.65 decades per layer.
(b) Forward signal: the second moment of each layer's output, Xavier vs He, for ReLU and tanh.
(c) The gain that keeps the signal's scale fixed, for ReLU, tanh, GELU and SiLU, at three input scales.
(d) Does the 18-layer net train? Two spirals with plain SGD, and with Adam, which rescales each gradient.
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
from nonlin.mlp import DeepMLP, decades_per_layer, forward_second_moments, grad_norms_per_layer, preserving_gain  # noqa: E402
from nonlin.stats import Results  # noqa: E402

CONFIGS = [("sigmoid", "xavier"), ("tanh", "xavier"), ("relu", "xavier"), ("relu", "he"),
           ("gelu", "he"), ("silu", "he")]


def part_a(res: Results, depth: int, width: int, seeds: list[int]) -> None:
    res.print(f"(a) Backward pass, {depth} layers of width {width}, Gaussian inputs, 10 random labels, batch 512, "
              f"mean over seeds {seeds}")
    res.print("dL/dz is the gradient at each pre-activation. dL/dW is the weight gradient. Decades per layer is the "
              "fitted slope of log10(norm), output side to input side.")
    res.print(f"{'activation':<9} {'init':<7} {'dL/dz dec/layer':>16} {'dL/dz first/last':>17} "
              f"{'dL/dW dec/layer':>16} {'dL/dW first/last':>17}")
    out = {}
    for act, init in CONFIGS:
        runs = []
        for s in seeds:
            torch.manual_seed(s)
            model = DeepMLP(width, width, 10, depth, act, init=init)
            x, y = torch.randn(512, width), torch.randint(10, (512,))
            runs.append(grad_norms_per_layer(model, x, y))
        row = {}
        for kind in ("signal", "weight"):
            norms = [r[kind] for r in runs]
            row[kind] = {
                "decades_per_layer": sum(decades_per_layer(n) for n in norms) / len(norms),
                "log10_first_over_last": sum(math.log10(n[0] / n[-1]) for n in norms) / len(norms),
                "log10_norm_by_layer": [sum(math.log10(n[i]) for n in norms) / len(norms) for i in range(depth)],
            }
        res.print(f"{act:<9} {init:<7} {row['signal']['decades_per_layer']:>16.3f} "
                  f"{row['signal']['log10_first_over_last']:>17.2f} {row['weight']['decades_per_layer']:>16.3f} "
                  f"{row['weight']['log10_first_over_last']:>17.2f}")
        out[f"{act}+{init}"] = row
    res.data["backward"] = out


def part_b(res: Results, depth: int, width: int) -> None:
    res.print(f"\n(b) Forward pass, {depth} layers of width {width}, unit-variance inputs: E[h^2] at layers 1, 6, 12, 18 "
              f"and the mean per-layer ratio")
    res.print(f"{'activation':<9} {'init':<7} {'layer1':>9} {'layer6':>9} {'layer12':>9} {'layer18':>9} {'ratio/layer':>12}")
    out = {}
    torch.manual_seed(0)
    x = torch.randn(4096, width)
    for act, init in [("relu", "xavier"), ("relu", "he"), ("tanh", "xavier"), ("tanh", "he")]:
        torch.manual_seed(1)
        m = forward_second_moments(DeepMLP(width, width, 10, depth, act, init=init), x)
        ratio = (m[-1] / m[0]) ** (1 / (depth - 1))
        res.print(f"{act:<9} {init:<7} {m[0]:>9.3g} {m[5]:>9.3g} {m[11]:>9.3g} {m[-1]:>9.3g} {ratio:>12.3f}")
        out[f"{act}+{init}"] = {"second_moment_by_layer": m, "ratio_per_layer": ratio}
    res.data["forward"] = out


def part_c(res: Results) -> None:
    res.print("\n(c) Gain g with Var(W) = g^2/fan_in that keeps E[z^2] = q fixed, z ~ N(0, q) (2M samples)")
    qs = [0.25, 1.0, 4.0]
    res.print(f"{'activation':<9} " + " ".join(f"{'q=' + str(q):>8}" for q in qs))
    out = {}
    for act in ["relu", "tanh", "gelu", "silu"]:
        gains = [preserving_gain(act, q) for q in qs]
        res.print(f"{act:<9} " + " ".join(f"{g:>8.3f}" for g in gains))
        out[act] = dict(zip(map(str, qs), gains))
    res.print(f"sqrt(2) = {math.sqrt(2):.3f}")
    res.data["preserving_gain"] = out


def part_d(res: Results, depth: int, steps: int, seeds: list[int]) -> None:
    res.print(f"\n(d) Two spirals, {depth} hidden layers of width 64, {steps} full-batch steps, "
              f"train accuracy, mean over seeds {seeds}")
    res.print(f"{'activation':<9} {'init':<7} {'SGD lr 0.1':>11} {'Adam lr 1e-3':>13}")
    x, y = two_spirals(2000, seed=0)
    out = {}
    for act, init in [("sigmoid", "xavier"), ("tanh", "xavier"), ("relu", "he")]:
        row = {}
        for name, make in [("sgd", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
                           ("adam", lambda p: torch.optim.Adam(p, lr=1e-3))]:
            accs = []
            for s in seeds:
                torch.manual_seed(s)
                model = DeepMLP(2, 64, 2, depth, act, init=init)
                opt = make(model.parameters())
                for _ in range(steps):
                    opt.zero_grad()
                    nn.functional.cross_entropy(model(x), y).backward()
                    opt.step()
                with torch.no_grad():
                    accs.append((model(x).argmax(1) == y).float().mean().item())
            row[name] = sum(accs) / len(accs)
        res.print(f"{act:<9} {init:<7} {row['sgd']:>11.3f} {row['adam']:>13.3f}")
        out[f"{act}+{init}"] = row
    res.data["train_deep"] = out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path, default=LAB / "results")
    args = p.parse_args()
    seeds = list(range(1 if args.quick else args.seeds))
    res = Results("02_gradient_decay", args.out)
    part_a(res, args.depth, 64 if args.quick else args.width, seeds)
    part_b(res, args.depth, 64 if args.quick else args.width)
    part_c(res)
    part_d(res, args.depth, 20 if args.quick else args.steps, seeds[:3])
    res.save()


if __name__ == "__main__":
    main()
