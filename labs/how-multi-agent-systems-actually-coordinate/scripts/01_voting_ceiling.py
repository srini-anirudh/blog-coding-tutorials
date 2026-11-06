"""The crowd is only wise if it disagrees: the correlation ceiling on majority voting.

    python labs/how-multi-agent-systems-actually-coordinate/scripts/01_voting_ceiling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mascoord import voting as V  # noqa: E402
from mascoord.report import Report  # noqa: E402

out = Report("01_voting_ceiling")
NS = [1, 3, 5, 9, 25, 101]

out("\n1. Independent search: P(at least one of N succeeds) = 1 - (1-p)^N, p = 0.4")
out("   " + "  ".join(f"N={n}: {V.p_any(n, 0.4):.3f}" for n in [1, 2, 3, 5, 10]))

out("\n2. Majority vote under a common-mode blind spot, p = 0.6 (closed form)")
out(f"{'rho':>6} " + " ".join(f"{'N=' + str(n):>7}" for n in NS) + f" {'ceiling':>8}")
for rho in [0.0, 0.2, 0.35, 0.5, 0.8]:
    out(f"{rho:>6.2f} " + " ".join(f"{V.vote_accuracy(n, 0.6, rho):>7.3f}" for n in NS)
        + f" {V.vote_ceiling(0.6, rho):>8.3f}")

out("\n3. Monte Carlo check, one agent at a time (200k trials, p = 0.6, rho = 0.35)")
out(f"{'N':>4} {'closed form':>12} {'simulated':>10} {'plurality*':>11} {'best-of-N':>10} {'best-of-N formula':>18}")
for n in [1, 3, 5, 9, 25]:
    sim = V.simulate(n, 0.6, 0.35)
    out(f"{n:>4} {V.vote_accuracy(n, 0.6, 0.35):>12.4f} {sim['strict_majority']:>10.4f} "
        f"{sim['plurality']:>11.4f} {sim['best_of_n']:>10.4f} {V.best_of_n_accuracy(n, 0.6, 0.35):>18.4f}")
out("   * plurality: wrong answers are spread over 50 values, so they rarely agree with each other.")

out("\n4. Verifier asymmetry: a weak agent, p = 0.3, rho = 0.2")
out(f"{'N':>4} {'majority':>9} {'best-of-N, perfect verifier':>28}")
for n in [1, 3, 9, 25]:
    out(f"{n:>4} {V.vote_accuracy(n, 0.3, 0.2):>9.3f} {V.best_of_n_accuracy(n, 0.3, 0.2):>28.3f}")
out("   Voting a below-chance agent makes it worse. A verifier still climbs toward its own ceiling,")
out(f"   which is 1 - rho(1 - p) = {V.vote_ceiling(0.3, 0.2):.3f}: shared blind spots cap it too.")

out("\n5. Quality versus diversity (a toy reading of Self-MoA), 6 agents, p_strong = 0.7")
out("   Mixing in a second family cuts shared blind spots from rho = 0.5 to rho = 0.3,")
out("   but the second family is weaker. Accuracy of the team:")
out(f"{'p_weak':>7} {'6 strong':>9} {'4 strong + 2 weak':>18} {'3 strong + 3 weak':>18}")
for pw in [0.65, 0.6, 0.55, 0.5]:
    out(f"{pw:>7.2f} {V.mixed_team_accuracy(6, 0, 0.7, pw, 0.5, 0.3):>9.3f} "
        f"{V.mixed_team_accuracy(4, 2, 0.7, pw, 0.5, 0.3):>18.3f} "
        f"{V.mixed_team_accuracy(3, 3, 0.7, pw, 0.5, 0.3):>18.3f}")
