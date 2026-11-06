"""Sections 04 and 08: the smooth gates, measured on a fine grid.

(a) GELU is the mean of an adaptive dropout mask: the Monte-Carlo estimate of E[x * Bernoulli(Phi(x))] converges
    to x * Phi(x) like 1/sqrt(draws) (Fig 04).
(b) Each curve's negative dip, and the largest value of f'(x) (Fig 05).
(c) How far apart the near-identical curves really are: GELU vs its tanh form, vs QuickGELU, vs SiLU.
(d) Swish x * sigmoid(beta x) runs from x/2 (beta -> 0) to ReLU (beta -> infinity).
(e) Every function here matches the PyTorch built-in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from nonlin import activations as A  # noqa: E402
from nonlin.stats import Results  # noqa: E402

GRID = torch.linspace(-8, 8, 1_600_001, dtype=torch.float64)  # step 1e-5


def argmin_on_grid(fn) -> tuple[float, float]:
    y = fn(GRID)
    i = int(torch.argmin(y))
    return GRID[i].item(), y[i].item()


def part_a(res: Results, seed: int) -> None:
    x = torch.linspace(-4, 4, 41, dtype=torch.float64)
    exact = A.gelu(x)
    gen = torch.Generator().manual_seed(seed)
    res.print("(a) Monte-Carlo mean of x * Bernoulli(Phi(x)) at 41 points in [-4, 4] vs exact GELU")
    res.print(f"{'draws':>7} {'max |error|':>12} {'max |error| * sqrt(draws)':>26}")
    out = {}
    for n in [1, 8, 64, 512, 4096, 32768]:
        err = (A.stochastic_gate_sample(x, n, gen) - exact).abs().max().item()
        res.print(f"{n:>7} {err:>12.4f} {err * n**0.5:>26.3f}")
        out[str(n)] = err
    res.data["monte_carlo_max_error"] = out


def part_b(res: Results) -> None:
    res.print("\n(b) Minimum of f(x) and maximum of f'(x), grid step 1e-5 on [-8, 8]")
    res.print(f"{'function':<11} {'argmin x':>9} {'min f':>9} {'argmax x':>9} {'max f-prime':>12}")
    out = {}
    for name in ["gelu", "gelu_tanh", "quick_gelu", "silu", "mish", "hard_swish", "elu", "relu"]:
        fn = A.get(name)
        xmin, fmin = argmin_on_grid(fn)
        d = A.derivative(fn, GRID)
        j = int(torch.argmax(d))
        res.print(f"{name:<11} {xmin:>9.4f} {fmin:>9.4f} {GRID[j].item():>9.4f} {d[j].item():>12.4f}")
        out[name] = {"argmin": xmin, "min": fmin, "argmax_derivative": GRID[j].item(), "max_derivative": d[j].item()}
    res.data["shape"] = out


def part_c(res: Results) -> None:
    res.print("\n(c) Largest pointwise gap on [-8, 8]")
    gelu = A.gelu(GRID)
    out = {}
    for name in ["gelu_tanh", "quick_gelu", "silu", "mish"]:
        gap = (A.get(name)(GRID) - gelu).abs()
        i = int(torch.argmax(gap))
        res.print(f"|GELU - {name:<10}|  max {gap[i].item():.6f} at x = {GRID[i].item():.3f}")
        out[name] = {"max_gap": gap[i].item(), "at_x": GRID[i].item()}
    res.data["gap_to_gelu"] = out


def part_d(res: Results) -> None:
    res.print("\n(d) Swish with beta: largest gap on [-8, 8] to x/2 and to ReLU")
    res.print(f"{'beta':>8} {'vs x/2':>10} {'vs ReLU':>10}")
    out = {}
    for beta in [0.01, 0.1, 1.0, 10.0, 100.0]:
        y = A.swish(GRID, beta)
        g_lin = (y - GRID / 2).abs().max().item()
        g_relu = (y - A.relu(GRID)).abs().max().item()
        res.print(f"{beta:>8} {g_lin:>10.4f} {g_relu:>10.4f}")
        out[str(beta)] = {"vs_half_x": g_lin, "vs_relu": g_relu}
    res.data["swish_beta"] = out


def part_e(res: Results) -> None:
    x = torch.linspace(-10, 10, 20001, dtype=torch.float64)
    pairs = {
        "sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": F.relu, "leaky_relu": F.leaky_relu, "elu": F.elu,
        "selu": F.selu, "relu6": F.relu6, "gelu": F.gelu, "gelu_tanh": lambda t: F.gelu(t, approximate="tanh"),
        "silu": F.silu, "mish": F.mish, "hard_swish": F.hardswish, "softplus": F.softplus,
    }
    worst = max((A.get(n)(x) - f(x)).abs().max().item() for n, f in pairs.items())
    res.print(f"\n(e) {len(pairs)} functions vs torch.nn.functional on [-10, 10], float64: worst gap {worst:.2e}")
    res.data["worst_gap_to_torch"] = worst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=LAB / "results")
    args = p.parse_args()
    res = Results("04_smooth_gates", args.out)
    part_a(res, 0)
    part_b(res)
    part_c(res)
    part_d(res)
    part_e(res)
    res.save()


if __name__ == "__main__":
    main()
