"""Run any OpenAI-compatible model through the harness on the toy repo.

    export OPENAI_BASE_URL=http://localhost:8000/v1   # vLLM, llama.cpp, Ollama, or api.openai.com/v1
    export OPENAI_API_KEY=...
    export HARNESS_MODEL=your-model-name
    python c10-agent-harness/scripts/run_openai_agent.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from harness import Agent, AgentConfig, EventLog, LocalSandbox, OpenAICompatModel, coding_tools  # noqa: E402
from harness.git_tools import init_repo  # noqa: E402
from harness.replay import format_event  # noqa: E402

TASK = "The test suite fails on median for even-length lists. Fix the bug and make all tests pass."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--log", default=str(HERE / "runs" / "openai" / "events.jsonl"))
    args = ap.parse_args()
    if "HARNESS_MODEL" not in os.environ:
        sys.exit("Set HARNESS_MODEL, OPENAI_BASE_URL and OPENAI_API_KEY first (see the docstring).")

    sandbox = LocalSandbox.from_copy(HERE / "examples" / "toy_repo")
    init_repo(sandbox.workspace)
    log = EventLog(args.log)
    agent = Agent(OpenAICompatModel(), coding_tools(), sandbox.workspace, sandbox=sandbox, log=log,
                  config=AgentConfig(max_steps=args.steps))
    try:
        result = agent.run(TASK)
        for e in log.events:
            print(format_event(e))
        print(f"\nstatus: {result.status} after {result.steps} steps")
        print(agent.tracer.table())
    finally:
        sandbox.cleanup()


if __name__ == "__main__":
    main()
