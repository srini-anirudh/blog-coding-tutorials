"""Section 7: per-item tool calls are Theta(n) round trips, a code action is Theta(1).

    python labs/inside-an-ai-agent-harness/scripts/code_action.py

Writes results/code_action.txt.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_lab.code_action import code_action_agent, json_tool_agent, make_repo, snapshot  # noqa: E402
from harness_lab.report import Report  # noqa: E402

TASK = "Rename the function parse_cfg to load_cfg in every Python file of this repository."


def main() -> None:
    rep = Report("code_action.txt")
    rep("Scripted agents, real files. Input tokens = sum over model calls of the context sent (C10 estimate).")
    rep(f"{'files':>6} {'matches':>8} | {'JSON calls':>10} {'JSON input':>11} {'JSON output':>11} | "
        f"{'code calls':>10} {'code input':>10} {'code output':>11} | {'input ratio':>11} {'same repo':>9}")
    for n in (5, 10, 20, 40, 80):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ra, rb = Path(a), Path(b)
            hits = make_repo(ra, n)
            make_repo(rb, n)
            tj, tc = json_tool_agent(ra, TASK), code_action_agent(rb, TASK)
            same = snapshot(ra) == snapshot(rb) and all("parse_cfg" not in s for s in snapshot(ra).values())
            rep(f"{n:>6} {hits:>8} | {tj.model_calls:>10} {tj.input_tokens:>11,} {tj.output_tokens:>11,} | "
                f"{tc.model_calls:>10} {tc.input_tokens:>10,} {tc.output_tokens:>11,} | "
                f"{tj.input_tokens / tc.input_tokens:>10.1f}x {str(same):>9}")
    rep()
    rep("JSON calls = 1 list + n reads + one write per match + 1 final answer. Code calls = 1 program + 1 final answer.")
    rep("File contents pass through the context in the JSON lane. In the code lane they stay in program variables.")
    rep.save()


if __name__ == "__main__":
    main()
