"""Sections "Post-training becomes an environment problem" and "What is still open": pass^k is not mean success.

Reads the per-task success probabilities that 03_credit.py saved, so run that first.

    python labs/the-making-of-an-ai-agent/scripts/04_reliability.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcraft.common import RESULTS, table, write_results
from agentcraft.reliability import estimate, exact_pass_at_k, exact_pass_hat_k

KS = [1, 2, 4, 8]


def rows_for(name: str, ps: list[float]) -> list[list]:
    est_at, est_hat = estimate(ps, n=16, k=8)
    return [[name, f"{sum(ps) / len(ps):.3f}"] + [f"{exact_pass_hat_k(ps, k):.3f}" for k in KS[1:]]
            + [f"{exact_pass_at_k(ps, 8):.3f}", f"{est_hat:.3f}", f"{est_at:.3f}"]]


def main():
    credit = json.loads((RESULTS / "03_credit.json").read_text())
    rows = []
    for method, noise in (("grpo", 0.0), ("gigpo", 0.0)):
        run = next(r for r in credit["runs"] if r["method"] == method and r["state_noise"] == noise and r["seed"] == 0)
        for h in run["history"]:
            if h["iter"] in (25, 50, 200):
                rows += rows_for(f"{method}, iteration {h['iter']}", h["per_task"])
    # Two agents with the same 65% mean success. One is 65% on every task; the other solves 65% of tasks always.
    rows += rows_for("same 0.65 on every task", [0.65] * 100)
    rows += rows_for("0.65 of tasks always, rest never", [1.0] * 65 + [0.0] * 35)
    text = ("Exact values from per-task success probabilities (seed 0 of 03_credit.py, 64 tasks), and two "
            "constructed agents.\nThe last two columns simulate 16 trials per task and apply the unbiased "
            "estimators, as a check.\n\n"
            + table(rows, ["agent", "mean success (pass^1)", "pass^2", "pass^4", "pass^8", "pass@8",
                           "pass^8 est. (n=16)", "pass@8 est. (n=16)"]))
    print(text)
    write_results("04_reliability", text, {"rows": rows})


if __name__ == "__main__":
    main()
