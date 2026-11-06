"""What each observation policy keeps from a giant test log.

A synthetic pytest log of 20,000 lines has one failing assertion placed at a random
position. For each policy we ask: how many tokens reach the model, and does the
assertion survive?

    python c10-agent-harness/benchmarks/bench_truncation.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.tokens import count_tokens  # noqa: E402
from harness.truncation import head_only, truncate_output  # noqa: E402

NEEDLE = "E       AssertionError: assert 3 == 2.5"


def make_log(n: int, pos: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = [f"tests/test_mod{rng.randrange(40)}.py::test_case_{i} PASSED [{i * 100 // n:>3}%]" for i in range(n)]
    lines[pos] = NEEDLE
    lines[pos - 1] = "    assert median([4, 1, 3, 2]) == 2.5"
    return "\n".join(lines)


def main() -> None:
    n, trials, budget = 20_000, 200, 120
    policies = {
        "no truncation": lambda t: t,
        "head only": lambda t: head_only(t, budget),
        "tail only": lambda t: "\n".join(t.splitlines()[-budget:]),
        "head + tail": lambda t: truncate_output(t, max_lines=budget, max_rescued=0).text,
        "head + tail + rescue": lambda t: truncate_output(t, max_lines=budget).text,
    }
    print(f"{n} line log, {trials} random failure positions, line budget {budget}\n")
    print(f"{'policy':<24}{'tokens shown':>14}{'lines shown':>13}{'assertion kept':>16}")
    for name, fn in policies.items():
        rng = random.Random(0)
        kept = tokens = shown = 0
        cost_trials = 0
        for t in range(trials):
            log = make_log(n, rng.randrange(1, n), t)
            out = fn(log)
            kept += NEEDLE in out
            if t < 20:  # counting tokens on 20k lines is the slow part; 20 samples are stable
                tokens += count_tokens(out)
                shown += len(out.splitlines())
                cost_trials += 1
        print(f"{name:<24}{tokens // cost_trials:>14}{shown // cost_trials:>13}{kept / trials:>15.1%}")


if __name__ == "__main__":
    main()
