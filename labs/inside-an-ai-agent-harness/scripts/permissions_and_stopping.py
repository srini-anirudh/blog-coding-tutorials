"""Sections 13 and 14: the approval ladder, vigilance decay, turn length and stopping.

    python labs/inside-an-ai-agent-harness/scripts/permissions_and_stopping.py

Writes results/permissions_and_stopping.txt. Everything here is a model with stated
parameters. The post's reported figures (93% approvals, 84% fewer prompts, 14 of 100
false finishes) are not reproduced; they come from production telemetry and a study.
"""

from __future__ import annotations

import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_lab.permissions import (ACTIONS, MODE_FIRST, MODES, SDK_ORDER, attention_ceiling, decide,  # noqa: E402
                                     expected_prompts_caught, mean_attention, simulate_catch, team_policy)
from harness_lab.report import Report  # noqa: E402
from harness_lab.stopping import mean_capped, mean_length, run_episode, sample_length, tail  # noqa: E402


def main() -> None:
    rep = Report("permissions_and_stopping.txt")
    pol = team_policy()
    rep("1. The ladder (hooks, deny, ask, mode, allow, callback) against checking the mode first")
    rep(f"{'action':>62} {'mode':>18} | {'SDK order':>18} | {'mode first':>18}")
    broken = 0
    for a in ACTIONS:
        for m in MODES:
            d1, s1 = decide(pol, a, m, SDK_ORDER)
            d2, s2 = decide(pol, a, m, MODE_FIRST)
            broken += (d1 != d2)
            mark = "  <- policy lost" if d1 != d2 else ""
            rep(f"{a.tool + ' ' + a.arg:>62} {m:>18} | {d1 + ' (' + s1 + ')':>18} | {d2 + ' (' + s2 + ')':>18}{mark}")
    rep(f"Cells where the order changes the decision: {broken} of {len(ACTIONS) * len(MODES)}. "
        "In each one a bypass mode overrides a rule or hook the team wrote down.")
    rep()

    a0, h = 1.0, 20
    rep(f"2. Vigilance decay a(n) = a0 2^(-n/h), a0 = {a0}, h = {h} prompts (illustrative values, not measured)")
    rep(f"{'N prompts':>10} {'mean a, closed':>15} {'simulated':>10} {'sum of a':>9}")
    for N in (5, 10, 20, 50, 100, 200):
        rep(f"{N:>10} {mean_attention(N, a0, h):>15.3f} {simulate_catch(N, a0, h, 50_000, seed=N):>10.3f} "
            f"{expected_prompts_caught(N, a0, h):>9.2f}")
    rep(f"Sum of attention saturates at {attention_ceiling(a0, h):.2f} prompts however many are shown.")
    for N in (50, 200):
        cut = round(N * 0.16)
        rep(f"Cutting {N} prompts by 84% to {cut}: chance of catching the one dangerous prompt "
            f"{mean_attention(N, a0, h):.3f} -> {mean_attention(cut, a0, h):.3f}")
    rep()

    rep("3. Turn length is geometric (100,000 sampled turns per q)")
    rep(f"{'q':>5} {'E[L]':>7} {'sampled':>8} {'P(L>40)':>9} {'sampled':>8} {'E[min(L,40)]':>13} {'sampled':>8}")
    rng = random.Random(0)
    for q in (0.80, 0.90, 0.95, 0.98):
        free = [sample_length(q, rng) for _ in range(100_000)]
        capped = [sample_length(q, rng, cap=40) for _ in range(100_000)]
        rep(f"{q:>5.2f} {mean_length(q):>7.1f} {statistics.mean(free):>8.2f} {tail(q, 40):>9.3f} "
            f"{sum(x > 40 for x in free) / len(free):>8.3f} {mean_capped(q, 40):>13.2f} {statistics.mean(capped):>8.2f}")
    rep()

    rep("4. Termination predicate: 3 plan items, p(progress) = 0.35, p(early done claim) = 0.08, cap 40, 10,000 tasks")
    for harness in ("naive", "conjunctive"):
        r = random.Random(1)
        eps = [run_episode(harness, r) for _ in range(10_000)]
        c = Counter(e.status for e in eps)
        rep(f"  {harness:>11}: solved {c['solved']:>5}, false victory {c['false victory']:>5}, "
            f"max turns {c['max turns']:>4}, mean steps {statistics.mean(e.steps for e in eps):.1f}")
    rep.save()


if __name__ == "__main__":
    main()
