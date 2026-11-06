"""Under an outcome reward, a step's credit is scaled by the chance that every other step went right.

    P(R=1 | step j right) - P(R=1 | step j wrong) = prod_{i != j} q_i

Checked in closed form and by Monte Carlo, then the post's two chart rollouts are
scored under the outcome reward and under step-level checks.

    python labs/rewarding-what-the-model-sees/scripts/credit_assignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrl.report import table, write  # noqa: E402


def step_credit_mc(n: int, q: float, samples: int, g: torch.Generator) -> float:
    steps = torch.rand(samples, n, generator=g) < q
    win = steps.all(1).float()
    right = steps[:, 0]
    return float(win[right].mean() - win[~right].mean())


def rollouts() -> str:
    """The chart has 2021 = 42 and 2023 = 57; the question asks for the growth, 15."""
    truth = (42, 57)
    cases = {"reads 41 and 56, subtracts to 15": (41, 56, 15),
             "reads 42 and 57, subtracts to 14": (42, 57, 14),
             "reads 42 and 57, subtracts to 15": (42, 57, 15)}
    rows = []
    for name, (a, b, ans) in cases.items():
        outcome = int(ans == truth[1] - truth[0])
        steps = [int(a == truth[0]), int(b == truth[1]), int(ans == b - a)]
        rows.append([name, f"{outcome} on every step", " / ".join(map(str, steps))])
    return table(rows, ["rollout", "outcome reward", "step checks: read 2021 / read 2023 / subtract"])


def main() -> None:
    g = torch.Generator().manual_seed(0)
    q, rows, data = 0.8, [], {}
    for n in range(1, 9):
        closed = q ** (n - 1)
        mc = step_credit_mc(n, q, 400_000, g)
        rows.append([n, f"{q ** n:.3f}", f"{closed:.3f}", f"{mc:.3f}"])
        data[n] = {"p_reward": q ** n, "credit_closed": closed, "credit_mc": mc}
    text = (f"Each step right with probability q = {q}, independently; reward only if all n steps are right.\n\n"
            + table(rows, ["steps n", "P(reward)", "credit of one step, closed form", "Monte Carlo (400k)"])
            + "\n\nThe post's chart example under the two rewards:\n\n" + rollouts())
    write("credit_assignment", text, {"q": q, "by_steps": data})


if __name__ == "__main__":
    main()
