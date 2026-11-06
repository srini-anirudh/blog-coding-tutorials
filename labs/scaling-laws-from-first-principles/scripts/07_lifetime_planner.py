"""Sections 8, 9 and 14: serving cost moves the optimum, repetition discounts tokens, and the planner.

Arithmetic on Hoffmann et al.'s published constants, as in the post's Figures 9, 10 and 16. If 05_fit_sweep.py has
run, part 1 is repeated with the law fitted to this lab's own sweep.

    python labs/scaling-laws-from-first-principles/scripts/07_lifetime_planner.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from scalelab.allocation import HOFFMANN, LossLaw, effective_tokens, lifetime_optimal, optimal, plan  # noqa: E402


def lifetime_table(law: LossLaw, target: float, demands: list[float], unit: str) -> None:
    base = lifetime_optimal(law, target, 0.0)
    print(f"   target loss {target:.3f}; train-optimal model N = {base[0]:.3g}, D = {base[1]:.3g}")
    print(f"   {'T_inf':>9s} {'N':>10s} {'D':>10s} {'D/N':>8s} {'N vs T=0':>9s} {'cost vs train-opt':>18s}")
    for t in demands:
        n, d, cost = lifetime_optimal(law, target, t)
        naive = 6 * base[0] * base[1] + 2 * base[0] * t
        print(f"   {t:9.1e} {n:10.3g} {d:10.3g} {d / n:8.0f} {n / base[0]:8.2f}x {cost / naive:17.2f}x")
    print(f"   (T_inf in {unit}; the last column is lifetime FLOPs relative to serving the train-optimal model)")


def main() -> None:
    print("# hardware: CPU (pure arithmetic)\n# command: python scripts/07_lifetime_planner.py\n")
    print("1. Lifetime cost 6 N D + 2 N T_inf at fixed loss (Hoffmann constants, Figure 9)")
    lifetime_table(HOFFMANN, 1.90, [0, 1e11, 1e12, 1e13, 1e14, 1e15], "tokens")
    print("   N falls monotonically as inference demand grows, and tokens per parameter rise.\n")

    fit = LAB / "results" / "fit_sweep.json"
    if fit.exists():
        law = LossLaw(**json.loads(fit.read_text())["parametric_total"])
        n0, d0 = optimal(law, 3e15)
        target = float(law(n0, d0))
        print("1b. The same calculation with the law fitted to this lab's TinyStories sweep (05_fit_sweep.py)")
        lifetime_table(law, target, [0, 1e9, 1e10, 1e11, 1e12], "tokens")
        print()

    print("2. Repeated data (Muennighoff et al. 2023 form, R* = 15.4, Figure 10)")
    U = 1e12
    print(f"   unique corpus U = {U:.0e} tokens")
    for ep in (1, 2, 4, 8, 16, 32, 64):
        de = effective_tokens(U, ep)
        print(f"   {ep:3d} epochs: {ep * U:8.1e} tokens seen, effective {de:8.2e} ({de / (ep * U):6.1%} of seen)")
    print(f"   ceiling (1 + R*) U = {16.4 * U:.2e}: no number of epochs gets past it in this functional form\n")

    print("3. The planner (Figure 16 defaults): 1,024 H100 for 720 h at 42% MFU, 6T unique tokens")
    for serve in (0.0, 1e13):
        p = plan(1024, 720, 989e12, 0.42, 6e12, inference_tokens=serve)
        print(f"   lifetime tokens served {serve:.0e}: budget C = {p.budget:.3e} FLOPs -> N* = {p.n / 1e9:.1f}B, "
              f"D* = {p.d / 1e12:.2f}T ({p.d / p.n:.0f} per parameter), {p.epochs:.2f} epochs of the corpus"
              + (f"; serving-aware N = {p.lifetime_n / 1e9:.1f}B" if p.lifetime_n else ""))
    print("   Peak 989 TFLOP/s is the vendor's dense BF16 figure. Constants are Hoffmann's, not yours.")


if __name__ == "__main__":
    main()
