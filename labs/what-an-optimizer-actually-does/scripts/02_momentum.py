"""Smoothing in time: the two momentum conventions, the averaging window, and kappa -> sqrt(kappa).

Claims checked:
  1. For a constant gradient the accumulator m = b m + g converges to g / (1 - b); the EMA converges to g.
  2. An EMA with factor b averages about 1 / (1 - b) gradients with mean age b / (1 - b).
  3. On a quadratic, gradient descent contracts by (k - 1)/(k + 1) per step. Heavy-ball momentum with
     b* = ((sqrt k - 1)/(sqrt k + 1))^2 contracts by (sqrt k - 1)/(sqrt k + 1).
  4. Below b* the iterate is over-damped and crawls; above it, it rings.
  5. Nesterov damps the overshoot of heavy ball at the same lr and b.

Run: python labs/what-an-optimizer-actually-does/scripts/02_momentum.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optlab import SGD
from optlab.common import Report

torch.set_default_dtype(torch.float64)


def quadratic(kappa, dim=50):
    """Eigenvalues log-spaced from 1 to kappa; start with unit error on every eigen-direction."""
    lam = torch.logspace(0, math.log10(kappa), dim)
    return lam, torch.ones(dim)


def run(lam, x0, lr, momentum=0.0, nesterov=False, steps=1000):
    """SGD in PyTorch's accumulator form on f = 0.5 sum(lam x^2). Returns the error norm per step and x."""
    x = torch.nn.Parameter(x0.clone())
    opt = SGD([x], lr, momentum=momentum, nesterov=nesterov)
    norms, slow = [], []
    for _ in range(steps):
        norms.append(x.norm().item())
        slow.append(x[0].item())  # the lowest-curvature coordinate
        x.grad = lam * x.detach()
        opt.step()
    return norms, slow


def rate(norms):
    """Per-step contraction measured over the second half of the run."""
    end = next((i for i, n in enumerate(norms) if n < 1e-200), len(norms)) - 1  # stop before underflow
    end = min(end, len(norms) - 1)
    t = end // 2
    return (norms[end] / norms[t]) ** (1 / (end - t))


def steps_to(norms, factor=1e-6):
    return next((i for i, n in enumerate(norms) if n < factor * norms[0]), None)


def sign_changes(xs):
    return sum(1 for a, b in zip(xs, xs[1:]) if a * b < 0)


def main():
    rep = Report("02_momentum")
    rep("## 1. Two conventions, constant gradient g = 1, b = 0.9, 200 steps")
    for ema in (False, True):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = SGD([p], lr=1.0, momentum=0.9, ema=ema)
        for _ in range(200):
            p.grad = torch.ones(1)
            opt.step()
        rep(f"{'EMA' if ema else 'accumulator':<12} m -> {opt.state[p]['m'].item():.4f}")

    rep("\n## 2. EMA window: weight (1-b) b^k on the gradient from k steps ago")
    for b in (0.9, 0.99, 0.999):
        k = torch.arange(100_000)
        w = (1 - b) * b**k
        rep(f"b={b:<6} sum of weights {w.sum():.4f}  1/(1-b) {1 / (1 - b):7.1f}  "
            f"mean age {(w * k).sum():7.1f}  (b/(1-b) = {b / (1 - b):.1f})")

    rep("\n## 3. Contraction per step on a 50-dim quadratic, eigenvalues in [1, k]")
    rep(f"{'kappa':>7} {'b*':>6} {'GD theory':>10} {'GD meas':>10} {'HB theory':>10} {'HB meas':>10} "
        f"{'GD steps to 1e-6':>17} {'HB steps to 1e-6':>17}")
    rows = {}
    for kappa in (100, 10_000):
        lam, x0 = quadratic(kappa)
        L, mu, sk = lam.max().item(), lam.min().item(), math.sqrt(kappa)
        gd_steps = 200_000 if kappa > 1000 else 5000
        gd, _ = run(lam, x0, lr=2 / (L + mu), steps=gd_steps)
        beta = ((sk - 1) / (sk + 1)) ** 2
        hb, _ = run(lam, x0, lr=4 / (math.sqrt(L) + math.sqrt(mu)) ** 2, momentum=beta, steps=3000)
        rows[kappa] = dict(beta_star=beta, gd=rate(gd), hb=rate(hb), gd_steps=steps_to(gd), hb_steps=steps_to(hb))
        rep(f"{kappa:>7} {beta:>6.3f} {(kappa - 1) / (kappa + 1):>10.6f} {rate(gd):>10.6f} "
            f"{(sk - 1) / (sk + 1):>10.6f} {rate(hb):>10.6f} {steps_to(gd)!s:>17} {steps_to(hb)!s:>17}")
    rep("(SGD's accumulator form: the heavy-ball lr is 4/(sqrt L + sqrt mu)^2 with m = b m + g.)")

    rep("\n## 4. kappa = 100, b varied, lr tuned for each b on a grid")
    lam, x0 = quadratic(100)
    rep(f"{'b':>6} {'best lr':>8} {'rate':>8} {'steps to 1e-6':>14} {'sign flips of slowest coord, 300 steps':>40}")
    sweep = {}
    for b in (0.3, 0.5, 0.669, 0.8, 0.9, 0.95):
        best = None
        for lr in torch.logspace(-3, 0, 61).tolist():
            norms, slow = run(lam, x0, lr, momentum=b, steps=300)
            if norms[-1] < norms[0] and (best is None or rate(norms) < best[1]):
                best = (lr, rate(norms), steps_to(norms), sign_changes(slow))
        sweep[b] = best
        rep(f"{b:>6} {best[0]:>8.4f} {best[1]:>8.4f} {best[2]!s:>14} {best[3]:>40}")
    rep("(with lr retuned, b below b* is over-damped: slower and no ringing; above b* it rings)")

    rep("\n## 5. Heavy ball vs Nesterov, kappa = 100, lr = 1/L, b = 0.9")
    for nest in (False, True):
        norms, slow = run(lam, x0, 1 / lam.max().item(), momentum=0.9, nesterov=nest, steps=2000)
        rep(f"{'Nesterov' if nest else 'heavy ball':<11} largest overshoot of slowest coord {-min(slow):.3f}  "
            f"sign flips {sign_changes(slow):>3}  steps to 1e-6 {steps_to(norms)}")
    rep.save({"contraction": rows, "beta_sweep": {str(k): v for k, v in sweep.items()}})


if __name__ == "__main__":
    main()
