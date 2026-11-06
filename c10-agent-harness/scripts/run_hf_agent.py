"""Run a real local model through the harness, on the same task as the scripted demo.

Qwen2.5-0.5B is tiny. Expect it to reach for tools and then lose the thread. The point
is to see what the harness does with a real, imperfect model, and to read the trace.

    python c10-agent-harness/scripts/run_hf_agent.py [--steps 8] [--device mps]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from harness import Agent, AgentConfig, ContextBuilder, EventLog, HFLocalModel, LocalSandbox, coding_tools  # noqa: E402
from harness.context import repo_section  # noqa: E402
from harness.git_tools import init_repo  # noqa: E402
from harness.replay import format_event  # noqa: E402

TASK = ("The test suite fails on median for even-length lists. Find the bug in stats_lib/core.py, "
        "fix it with edit_file, and run run_tests to check.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--preset", default="small-model")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--log", default=str(HERE / "results" / "hf_events.jsonl"))
    args = ap.parse_args()
    torch.set_num_threads(2)

    model = HFLocalModel(args.model, device=args.device, max_new_tokens=args.max_new_tokens)
    print(f"model: {args.model} on {model.device}, {sum(p.numel() for p in model.model.parameters()) / 1e6:.0f}M parameters")
    sandbox = LocalSandbox.from_copy(HERE / "examples" / "toy_repo")
    init_repo(sandbox.workspace)
    log = EventLog(args.log)
    agent = Agent(model, coding_tools(args.preset), sandbox.workspace, sandbox=sandbox, log=log,
                  context=ContextBuilder(sections={"repo": repo_section(300)}),
                  config=AgentConfig(max_steps=args.steps, context_budget=6000, keep_recent_turns=3))
    try:
        result = agent.run(TASK)
        for e in log.events:
            print(format_event(e))
        print(f"\nstatus: {result.status} after {result.steps} steps")
        print(agent.tracer.table())
        core = (sandbox.workspace / "stats_lib" / "core.py").read_text()
        print("\nbug fixed:", "(s[mid - 1] + s[mid]) / 2" in core or "% 2" in core)
        tests = [e.data["status"] for e in log.of_type("TestResult")]
        print("test runs:", tests)
    finally:
        sandbox.cleanup()


if __name__ == "__main__":
    main()
