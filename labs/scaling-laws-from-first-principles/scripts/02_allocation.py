"""Sections 1, 5, 6 and 13: the allocation problem, solved in closed form and checked by brute force.

Uses Hoffmann et al.'s published Approach 3 constants, exactly as the post's figures do. This is arithmetic on
published numbers, not a reproduction of the paper's experiments.

    python labs/scaling-laws-from-first-principles/scripts/02_allocation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scalelab.allocation import HOFFMANN as H  # noqa: E402
from scalelab.allocation import LossLaw, optimal, optimal_by_search, shortfalls  # noqa: E402


def main() -> None:
    print("# hardware: CPU (pure arithmetic)\n# command: python scripts/02_allocation.py\n")
    print(f"Loss law: L = {H.E} + {H.A} N^-{H.alpha} + {H.B} D^-{H.beta}  (Hoffmann et al. 2022, Approach 3)")
    print(f"Predicted exponents: N* ~ C^{H.a:.4f}, D* ~ C^{H.b:.4f}, sum {H.a + H.b:.4f}\n")

    print("1. Closed form against a brute-force search along each budget curve")
    print(f"{'C':>9s} {'N* closed':>11s} {'N* search':>11s} {'D*':>10s} {'D*/N*':>7s} {'L*':>7s} {'aT_N / bT_D':>12s}")
    for C in (1e19, 1e21, 1e23, 5.76e23, 3.8e25):
        n, d = optimal(H, C)
        ns, _ = optimal_by_search(H, C)
        tn, td = shortfalls(H, n, d)
        print(f"{C:9.2e} {n:11.4e} {ns:11.4e} {d:10.3e} {d / n:7.1f} {float(H(n, d)):7.4f} "
              f"{H.alpha * tn / (H.beta * td):12.6f}")
    print("The last column is the section 5 invariant alpha*T_N = beta*T_D: it is 1 at every budget.\n")

    n_g, d_g = optimal(H, 5.76e23)
    print("2. Gopher's budget, 5.76e23 FLOPs (the post's Approximation 00)")
    print(f"   these constants recommend N = {n_g / 1e9:.1f}B at {d_g / n_g:.0f} tokens per parameter;")
    print(f"   Chinchilla trained 70B on 1.4T (20 per parameter). L(70B, 1.4T) = {float(H(70e9, 1.4e12)):.4f}, "
          f"L at the recommendation = {float(H(n_g, d_g)):.4f}\n")

    print("3. How flat is the valley? Loss on the 1e23 budget curve, relative to its minimum")
    n0, _ = optimal(H, 1e23)
    l0 = float(H(n0, 1e23 / (6 * n0)))
    for f in (1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8):
        n = n0 * f
        print(f"   N = {f:6.3f} x N*: loss {float(H(n, 1e23 / (6 * n))):.4f}  (+{float(H(n, 1e23 / (6 * n))) - l0:.4f})")
    print("   A factor of 2 either way costs under 0.01 nats.")
    for C in (1e19, 1e23, 1e25):
        n0, _ = optimal(H, C)
        l0 = float(H(n0, C / (6 * n0)))
        lo, hi = (float(H(n0 * f, C / (6 * n0 * f))) - l0 for f in (1 / 4, 4))
        print(f"   C = {C:.0e}: N*/4 costs +{lo:.4f}, 4 N* costs +{hi:.4f} -> "
              f"{'too small' if lo > hi else 'too big'} is the worse mistake under these constants")
    print()

    print("4. Tokens per parameter drifts as C^((alpha-beta)/(alpha+beta))")
    drift = (H.alpha - H.beta) / (H.alpha + H.beta)
    r1 = optimal(H, 1e21)[1] / optimal(H, 1e21)[0]
    r2 = optimal(H, 1e25)[1] / optimal(H, 1e25)[0]
    print(f"   exponent {drift:.4f}; D*/N* goes from {r1:.1f} at 1e21 to {r2:.1f} at 1e25 (x{r2 / r1:.3f}, "
          f"10^(4*{drift:.4f}) = {10 ** (4 * drift):.3f})")
    print(f"   Aside 01: anchored at 20 for Chinchilla's 5.8e23 and moved to 3.8e25: "
          f"20 x {3.8e25 / 5.8e23:.0f}^{drift:.3f} = {20 * (3.8e25 / 5.8e23) ** drift:.1f}  (Meta measured 41)")
    eq = LossLaw(H.E, H.A, H.B, 0.31, 0.31)
    rs = [optimal(eq, C)[1] / optimal(eq, C)[0] for C in (1e19, 1e22, 1e25)]
    print(f"   with alpha = beta = 0.31 the ratio is {rs[0]:.2f}, {rs[1]:.2f}, {rs[2]:.2f} at 1e19, 1e22, 1e25: "
          "constant, the only case where a fixed '20' can exist.\n")

    print("5. Constants move the intercept, only alpha/beta moves the slope")
    for name, law in (("published", H), ("A x 3", LossLaw(H.E, 3 * H.A, H.B, H.alpha, H.beta)),
                      ("B / 3, E - 0.3", LossLaw(H.E - 0.3, H.A, H.B / 3, H.alpha, H.beta))):
        cs = np.logspace(19, 25, 7)
        ns = [optimal(law, c)[0] for c in cs]
        slope = np.polyfit(np.log(cs), np.log(ns), 1)[0]
        print(f"   {name:15s} N*(1e23) = {optimal(law, 1e23)[0]:.3e}  fitted slope d log N*/d log C = {slope:.4f}")
    print()

    print("6. Section 13: tokens per parameter for an MoE depends on which N")
    for name, tot, act, toks in (("DeepSeek-V3", 671e9, 37e9, 14.8e12),):
        print(f"   {name}: {toks / tot:.0f} tokens per total parameter, {toks / act:.0f} per active parameter")


if __name__ == "__main__":
    main()
