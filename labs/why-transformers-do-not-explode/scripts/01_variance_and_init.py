"""Parts I and II: one matmul multiplies variance by n_in * sigma_w^2, and L of them multiply it L times.

    python labs/why-transformers-do-not-explode/scripts/01_variance_and_init.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stablelab.common import Report, seed_all  # noqa: E402
from stablelab.signal import log_variance_walk, nonlinearity_gain, plain_stack  # noqa: E402


def main() -> None:
    seed_all(0)
    r = Report("01_variance_and_init")

    r("1. One matmul: Var(y) = n_in * sigma_w^2 * Var(x). sigma_x = 1, 4096 samples per row.")
    rows = []
    for n_in in (64, 256, 1024):
        for sw in (0.5 / math.sqrt(n_in), 1 / math.sqrt(n_in), 2 / math.sqrt(n_in)):
            x = torch.randn(4096, n_in, dtype=torch.float64)
            w = torch.randn(256, n_in, dtype=torch.float64) * sw
            rows.append([n_in, round(sw * math.sqrt(n_in), 2), n_in * sw**2, (x @ w.T).var().item()])
    r.table(["n_in", "sqrt(n_in)*sigma_w", "predicted Var(y)", "measured Var(y)"], rows)
    r("Only the product sqrt(n_in) * sigma_w matters: the same gain gives the same variance at every fan-in.")
    r()

    r("2. Nonlinearity gain gamma = 1 / E[phi(z)^2], z ~ N(0,1), Monte Carlo with 2M samples,")
    r("   against torch.nn.init.calculate_gain(name)^2.")
    rows = []
    for act in ("relu", "tanh", "linear"):
        mc = nonlinearity_gain(act)
        rows.append([act, mc, torch.nn.init.calculate_gain(act) ** 2])
    r.table(["phi", "gamma (Monte Carlo)", "calculate_gain^2"], rows)
    r("ReLU matches exactly (gamma = 2). For tanh, PyTorch's 5/3 (squared: 2.78) is a heuristic, not 1/E[tanh(z)^2].")
    r()

    r("3. Sixty plain layers, width 512, batch 256, float64. Second moment of the pre-activations at layer 60")
    r("   (forward) and of the gradient at the network input (backward, unit-variance gradient at the top).")
    rows = []
    stacks = {}
    for act in ("relu", "tanh", "linear"):
        for scheme in ("naive", "xavier", "kaiming"):
            out = plain_stack(60, 512, scheme, act)
            stacks[(act, scheme)] = out
            fwd, bwd = out["fwd"], out["bwd"]
            per_layer = (fwd[-1] / fwd[0]) ** (1 / 59)
            rows.append([act, scheme, f"{fwd[-1]:.3e}", f"{bwd[0]:.3e}", round(per_layer, 4)])
    r.table(["phi", "init", "E[z^2] at layer 60", "E[g^2] at input", "per-layer fwd gain"], rows)
    k, x = stacks[("relu", "kaiming")]["fwd"], stacks[("relu", "xavier")]["fwd"]
    r(f"ReLU: Xavier's per-layer gain is {(x[-1] / x[0]) ** (1 / 59):.4f}, Kaiming's is {(k[-1] / k[0]) ** (1 / 59):.4f}."
      " The factor of 2 compounds to 2^-59 over the stack.")
    t = stacks[("tanh", "kaiming")]
    r(f"tanh with Kaiming: the forward pass survives (tanh saturates at 1), but the input gradient has second moment"
      f" {t['bwd'][0]:.3g}. Forward health does not imply backward health.")
    r()

    r("4. Criticality is not enough: 64 critical linear networks (W ~ N(0, 1/n), mean gain exactly 1).")
    r("   Std across networks of ln(||x_L||^2 / ||x_0||^2). A zero-drift random walk spreads like sqrt(L).")
    rows = []
    for width in (64, 512):
        walk = log_variance_walk(100, width, n_nets=64)
        steps = np.diff(walk, axis=1)
        s = steps.std()
        for depth in (25, 50, 100):
            rows.append([width, depth, walk[:, depth].mean(), walk[:, depth].std(), s * math.sqrt(depth)])
    r.table(["width", "L", "mean ln-var", "std ln-var", "s*sqrt(L) prediction"], rows)
    r("s is the per-layer std of the log-gain, measured from the same runs. Narrower layers jitter more.")
    r("The mean is not zero: E[ln gain] is about -1/n per layer even though E[gain] = 1 (Jensen's inequality),")
    r("so L = 100 at width 64 should drift by roughly -100/64 = -1.6 in ln-variance (the table shows the measured value).")
    r.save()


if __name__ == "__main__":
    main()
