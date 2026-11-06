"""Six optimizers on the post's 2-D landscapes, each at its own best learning rate.

Claim checked: the ranking is not fixed. Adaptive (diagonal) methods win when the curvature is
axis-aligned and lose the advantage when it is not, because a diagonal preconditioner cannot see
rotated curvature. A rotated copy of the bowl isolates that one effect.

Run: python labs/what-an-optimizer-actually-does/scripts/01_landscapes.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optlab import SGD, Adam, Muon, RMSProp
from optlab.common import Report
from optlab.landscapes import grad, landscapes

OPTIMIZERS = {
    "SGD": lambda p, lr: SGD(p, lr),
    "Momentum": lambda p, lr: SGD(p, lr, momentum=0.9),
    "Nesterov": lambda p, lr: SGD(p, lr, momentum=0.9, nesterov=True),
    "RMSProp": lambda p, lr: RMSProp(p, lr, beta2=0.99),
    "AdamW": lambda p, lr: Adam(p, lr),
    # A 2-vector viewed as a 1x2 matrix: orthogonalizing it gives (about) the unit vector m / |m|.
    "Muon-style": lambda p, lr: Muon(p, lr, rms_scale=1 / math.sqrt(2)),
}
LRS = [10 ** (k / 4) for k in range(-12, 1)]  # 1e-3 ... 1, four per decade
TOL = 1e-3  # "converged" means f - f* < TOL
BUDGET = {"rosenbrock valley": 3000}  # steps; every other landscape gets 300
SEEDS = 5  # for the noisy landscape


def run(land, make, lr, steps, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    p = torch.nn.Parameter(torch.tensor(land.start, dtype=torch.float32))
    opt = make([p], lr)
    gaps = []
    for _ in range(steps):
        g, f = grad(land, p, gen)
        gaps.append(f - land.f_star)
        if not math.isfinite(f) or f > 1e6:
            return gaps + [math.inf] * (steps - len(gaps))
        p.grad = g
        opt.step()
    return gaps


def score(land, gaps):
    """Steps to reach TOL (noise-free), or the mean gap over the last 50 steps (noisy)."""
    if land.noise:
        return sum(gaps[-50:]) / 50
    hit = next((i for i, x in enumerate(gaps) if x < TOL), None)
    return hit if hit is not None else math.inf


def main():
    rep = Report("01_landscapes")
    rep(f"best lr per optimizer from {len(LRS)} values in [1e-3, 1]; tolerance f - f* < {TOL}")
    table = {}
    for land in landscapes():
        steps = BUDGET.get(land.name, 300)
        rep(f"\n## {land.name}  (start {land.start}, {steps} steps)")
        rep(f"{'optimizer':<11} {'best lr':>8} {'result':>22}")
        table[land.name] = {}
        for name, make in OPTIMIZERS.items():
            best = (math.inf, None)
            for lr in LRS:
                seeds = range(SEEDS) if land.noise else [0]
                s = sum(score(land, run(land, make, lr, steps, sd)) for sd in seeds) / len(seeds)
                best = min(best, (s, lr), key=lambda t: t[0])
            s, lr = best
            if land.noise:
                text = f"mean gap {s:.2e} (last 50)"
            else:
                text = f"{int(s)} steps" if s < math.inf else f"not in {steps} steps"
            rep(f"{name:<11} {lr if lr else float('nan'):>8.4f} {text:>22}")
            table[land.name][name] = {"best_lr": lr, "score": None if s == math.inf else s}
    rep.save({"tolerance": TOL, "results": table})


if __name__ == "__main__":
    main()
