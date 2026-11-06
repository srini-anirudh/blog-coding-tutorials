"""Partial credit and its price: five checkpoints, one goal state, and an agent that satisfies the checkpoints.

    python labs/the-agent-evaluation-playbook/scripts/02_partial_credit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenteval.repo import AGENTS, checkpoints, partial_score, run_agent, terminal, suite_unchanged  # noqa: E402
from agenteval.report import Report  # noqa: E402


def main() -> None:
    r = Report("02_partial_credit")
    r.add("task: fix mean() so the three original tests pass. Agents are scripted.")
    r.add()
    runs = {name: run_agent(name) for name in AGENTS}
    names = list(checkpoints(*runs["fixer"][:2]))
    rows = []
    for name, (before, after, msg) in runs.items():
        cp = checkpoints(before, after)
        rows.append([name] + ["x" if cp[n] else "." for n in names] +
                     ["PASS" if terminal(after) else "fail", "yes" if suite_unchanged(before, after) else "NO", msg])
    r.add("1. Checkpoints (x = satisfied) against the terminal goal")
    r.table(["agent"] + names + ["goal state", "tests intact", "final message"], rows)

    r.add("2. Partial score R = w_T * 1[goal] + (1 - w_T)/5 * sum of checkpoints, as w_T moves to the goal")
    ws = [0.0, 1 / 6, 0.5, 0.8, 0.95, 1.0]
    rows = []
    for name, (before, after, _) in runs.items():
        cp, term = checkpoints(before, after), terminal(after)
        rows.append([name] + [partial_score(cp, term, w) for w in ws])
    mean_partial = [sum(row[i + 1] for row in rows) / len(rows) for i in range(len(ws))]
    mean_term = sum(terminal(a) for _, a, _ in runs.values()) / len(runs)
    rows.append(["suite mean"] + mean_partial)
    rows.append(["terminal-only mean"] + [mean_term] * len(ws))
    r.table(["agent"] + [f"w_T={w:.2f}" for w in ws], rows, fmt="{:.2f}")
    r.add("w_T = 1/6 is 'the goal is just one more checkpoint'. The shortcut agent outscores the half-fix at every")
    r.add("w_T below 1 and ties it at w_T = 1. The suite mean meets the terminal score only at w_T = 1.")
    r.save()


if __name__ == "__main__":
    main()
