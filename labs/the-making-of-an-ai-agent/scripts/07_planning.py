"""Section "Planning versus execution": plan-then-act, ReAct and a hybrid in a folder another writer edits.

    python labs/the-making-of-an-ai-agent/scripts/07_planning.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcraft.common import table, write_results
from agentcraft.planning import AGENTS, run

Q, LENGTHS, EPISODES = 0.05, [2, 4, 8, 16, 32], 20000


def main():
    rows, data = [], []
    for L in LENGTHS:
        r = {a: run(a, L, Q, EPISODES, seed=L) for a in AGENTS}
        data.append({"files": L, **r, "predicted_plan_success": (1 - Q) ** L})
        rows.append([L, (1 - Q) ** L] + [x for a in AGENTS for x in (r[a]["success"], r[a]["calls"])])
    half_life = math.log(0.5) / math.log(1 - Q)
    text = (f"Another writer moves one unread file with probability q = {Q} after every tool call. {EPISODES} "
            f"episodes per row.\nPredicted plan-then-act success is (1 - q)^L. Plan half-life = ln 0.5 / "
            f"ln(1 - q) = {half_life:.1f} steps.\n\n"
            + table(rows, ["files L", "(1-q)^L", "plan-then-act success", "calls", "react success", "calls",
                           "hybrid success", "calls"]))
    print(text)
    write_results("07_planning", text, {"q": Q, "half_life_steps": half_life, "rows": data})


if __name__ == "__main__":
    main()
