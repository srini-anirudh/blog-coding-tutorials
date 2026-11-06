"""Sections 3 and 6 on synthetic data, where the true answer is known: does each fitting procedure recover it?

1. A power law with a floor (Figure 4): fit on a window, extrapolate, and see what a wrong floor does.
2. The IsoFLOP procedure (Figure 7, Approach 2) on losses drawn from the Hoffmann law with 0.5% noise.
3. The parametric fit (Approach 3) on the same runs.

    python labs/scaling-laws-from-first-principles/scripts/03_fit_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scalelab.allocation import HOFFMANN as H  # noqa: E402
from scalelab.allocation import optimal  # noqa: E402
from scalelab.fit import fit_parametric, isoflop_minimum, power_law, power_law_with_floor  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    print("# hardware: CPU (numpy and torch float64)\n# command: python scripts/03_fit_synthetic.py\n")

    print("1. L(N) = 1.69 + 406.4 N^-0.34 with 1.5% multiplicative noise, fit on N = 1e7 .. 10^9.5")
    n = np.logspace(7, 9.5, 12)
    y = (H.E + H.A * n ** -H.alpha) * np.exp(rng.normal(0, 0.015, n.size))
    E, A, al = power_law_with_floor(n, y)
    A0, al0 = power_law(n, y)
    target = 1e12
    truth = H.E + H.A * target ** -H.alpha
    print(f"   fit with a floor : E = {E:.3f}, A = {A:.1f}, alpha = {al:.3f}")
    print(f"   fit without floor: alpha = {al0:.3f} (the floor is absorbed into a shallower slope)")
    print(f"   extrapolated to N = 1e12 (2.5 decades out). True loss {truth:.4f}")
    print(f"     floor fitted       -> {E + A * target ** -al:.4f}")
    print(f"     no floor           -> {A0 * target ** -al0:.4f}")
    for guess in (1.2, 1.5, 1.69, 1.9):
        Ag, alg = power_law(n, np.clip(y - guess, 1e-6, None))
        print(f"     floor fixed at {guess:4.2f} -> {guess + Ag * target ** -alg:.4f}  (alpha {alg:.3f})")
    print("   The fit window cannot pin the floor; the extrapolation inherits whatever floor you assume.\n")

    print("2. IsoFLOP sweep on the Hoffmann law, 0.5% noise, 9 sizes per budget spanning 1/8 .. 8 x N*")
    budgets = np.logspace(18, 21, 4)
    rows, mins = [], []
    for C in budgets:
        n0, _ = optimal(H, C)
        ns = n0 * np.logspace(-0.9, 0.9, 9)
        ds = C / (6 * ns)
        ls = H(ns, ds) * np.exp(rng.normal(0, 0.005, ns.size))
        rows += list(zip(ns, ds, ls))
        n_opt, _ = isoflop_minimum(ns, ls)
        mins.append(n_opt)
        print(f"   C = {C:.0e}: valley minimum N = {n_opt:.3e}  (true N* = {n0:.3e})")
    a_fit = np.polyfit(np.log(budgets), np.log(mins), 1)[0]
    print(f"   fitted exponent a = {a_fit:.3f}; true beta/(alpha+beta) = {H.a:.3f}")
    n_pred = np.exp(np.polyval(np.polyfit(np.log(budgets), np.log(mins), 1), np.log(1e23)))
    print(f"   extrapolated to 1e23 (2 decades out): N = {n_pred:.3e}, true {optimal(H, 1e23)[0]:.3e}\n")

    print("3. Parametric fit (Approach 3) on the same 36 runs: Huber loss on log L, grid of 2,000 starts")
    N, D, L = (np.array(c) for c in zip(*rows))
    law, obj = fit_parametric(N, D, L)
    print(f"   fitted: E = {law.E:.3f}  A = {law.A:.1f}  B = {law.B:.1f}  alpha = {law.alpha:.3f}  beta = {law.beta:.3f}")
    print(f"   true  : E = {H.E:.3f}  A = {H.A:.1f}  B = {H.B:.1f}  alpha = {H.alpha:.3f}  beta = {H.beta:.3f}")
    print(f"   implied a = {law.a:.3f} (true {H.a:.3f}); N*(1e23) = {optimal(law, 1e23)[0]:.3e}")
    print("   A, B and E trade off against each other; the exponent ratio, which sets the policy, is better pinned.")


if __name__ == "__main__":
    main()
