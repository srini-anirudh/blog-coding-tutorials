"""Amdahl, Brooks, Coase: speedup, then the turn.

    python labs/how-multi-agent-systems-actually-coordinate/scripts/03_amdahl.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mascoord import scaling as S  # noqa: E402
from mascoord.report import Report  # noqa: E402

out = Report("03_amdahl")
f, kappa, alpha = 0.8, 0.01, 1.6

out(f"\n1. Amdahl only, f = {f}: limit 1/(1-f) = {S.amdahl_limit(f):.1f}x")
out("   " + "  ".join(f"N={n}: {S.amdahl_speedup(n, f):.2f}x" for n in [1, 2, 4, 8, 16, 64, 1024]))

out(f"\n2. With coordination, T(N) = (1-f) + f/N + kappa N^alpha, kappa = {kappa}, alpha = {alpha}")
out(f"{'N':>4} {'Amdahl':>8} {'with coord':>11} {'T(N)':>7}")
for n in [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32]:
    out(f"{n:>4} {S.amdahl_speedup(n, f):>8.2f} {S.real_speedup(n, f, kappa, alpha):>11.2f} "
        f"{S.wall_time(n, f, kappa, alpha):>7.3f}")
nb = S.best_n(f, kappa, alpha)
out(f"   best integer N* = {nb}, speedup {S.real_speedup(nb, f, kappa, alpha):.2f}x "
    f"(continuous optimum {S.best_n_continuous(f, kappa, alpha):.2f})")
out(f"   slower than one agent from N = {S.break_even_n(f, kappa, alpha)}")

out("\n3. Where the optimum sits as the coordination term changes")
out(f"{'f':>5} {'kappa':>7} {'alpha':>6} {'N*':>4} {'best speedup':>13} {'slower than solo at':>20}")
for f2, k2, a2 in [(0.8, 0.01, 1.0), (0.8, 0.01, 1.6), (0.8, 0.01, 2.0), (0.95, 0.01, 1.6),
                   (0.95, 0.002, 1.6), (0.5, 0.01, 1.6), (0.8, 0.0, 1.6)]:
    nb = S.best_n(f2, k2, a2)
    be = S.break_even_n(f2, k2, a2) if k2 > 0 else None
    out(f"{f2:>5} {k2:>7} {a2:>6} {nb:>4} {S.real_speedup(nb, f2, k2, a2):>12.2f}x {str(be):>20}")
out("   kappa = 0 is pure Amdahl: the search hits its upper bound of 256 agents.")
