"""End-to-end demo: a scripted agent fixes a failing test in a toy repo.

The model is scripted (deterministic), but everything around it is real: the tools
run, the tests run in the sandbox, the edits hit disk, the permission layer decides,
and every action lands in the JSONL event log.

    python c10-agent-harness/scripts/demo_fix_test.py [--log results/demo_events.jsonl]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from harness import (Agent, AgentConfig, EventLog, LocalSandbox, PermissionPolicy,  # noqa: E402
                     ScriptedModel, call, coding_tools)
from harness.git_tools import init_repo  # noqa: E402
from harness.messages import reset_call_ids  # noqa: E402
from harness.replay import format_event  # noqa: E402

TASK = "The test suite fails on median for even-length lists. Fix the bug and make all tests pass."

FIX_OLD = "    mid = len(s) // 2\n    return s[mid]"
FIX_NEW = ("    mid = len(s) // 2\n"
           "    if len(s) % 2 == 0:\n"
           "        return (s[mid - 1] + s[mid]) / 2\n"
           "    return s[mid]")

SCRIPT = [
    "I'll write a plan before touching anything.\n" + call("update_plan", items=[
        "[>] reproduce the failure", "[ ] find the median code", "[ ] fix it", "[ ] run the full suite"]),
    "Make sure pytest is installed.\n" + call("run_command", command="pip install pytest"),
    "Installing was denied; run_tests should work without it.\n" + call("run_tests"),
    # A realistic slip: a missing quote. The harness answers with a parse error.
    'Two failures, both about median. Find it.\n<tool_call>{"name": "search", "arguments": {"pattern": "def median}}</tool_call>',
    "Fix the JSON and search again.\n" + call("search", pattern="def median"),
    "Read the implementation and the failing test together.\n"
    + call("read_file", path="stats_lib/core.py") + "\n" + call("read_file", path="tests/test_stats.py", start=14, end=16),
    "For even n it returns the upper middle element. Update the plan, then fix.\n" + call("update_plan", items=[
        "[x] reproduce the failure", "[x] find the median code", "[>] fix it", "[ ] run the full suite"]),
    # A second slip: quoting `len(xs)` from memory instead of the file's `len(s)`.
    "Average the two middle elements when n is even.\n"
    + call("edit_file", path="stats_lib/core.py", search="    mid = len(xs) // 2\n    return s[mid]", replace=FIX_NEW),
    "The search text was stale. Use the exact lines shown.\n"
    + call("edit_file", path="stats_lib/core.py", search=FIX_OLD, replace=FIX_NEW),
    "Verify.\n" + call("run_tests"),
    "All green. Review the final diff.\n" + call("git_diff"),
    call("update_plan", items=["[x] reproduce the failure", "[x] find the median code", "[x] fix it",
                               "[x] run the full suite"]),
    "Fixed `median` in `stats_lib/core.py`: for an even number of values it now returns the mean of "
    "the two middle values instead of the upper one. All 6 tests pass.",
]


def demo_approver(name: str, arguments: dict, risk) -> bool:
    print(f"  [approval requested] {name} {arguments} risk={risk.name} -> denied (demo policy)")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(HERE / "results" / "demo_events.jsonl"))
    ap.add_argument("--budget", type=int, default=4600, help="context budget in estimated tokens")
    args = ap.parse_args()

    reset_call_ids()
    sandbox = LocalSandbox.from_copy(HERE / "examples" / "toy_repo")
    init_repo(sandbox.workspace)
    log = EventLog(args.log)
    agent = Agent(ScriptedModel(SCRIPT), coding_tools(), sandbox.workspace, sandbox=sandbox,
                  policy=PermissionPolicy(approver=demo_approver), log=log,
                  config=AgentConfig(context_budget=args.budget, keep_recent_turns=3))
    try:
        result = agent.run(TASK)
        for e in log.events:
            print(format_event(e))
        print(f"\nstatus: {result.status} after {result.steps} steps")
        print(agent.tracer.table())
        print(f"\nevent log: {Path(args.log).name} ({len(log.events)} events)")
    finally:
        sandbox.cleanup()


if __name__ == "__main__":
    main()
