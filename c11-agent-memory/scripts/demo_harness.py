"""Memory inside the real C10 Agent: two sessions, a fresh Agent each, one AgentMemory.

    python c11-agent-memory/scripts/demo_harness.py

Prints the context breakdown the harness logs and the tool trace of each task.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from agentmem import AgentMemory, MemoryEvent  # noqa: E402
from agentmem.integration import attach_memory  # noqa: E402
from harness import Agent, AgentConfig, ScriptedModel, call, tool, ToolRegistry  # noqa: E402


@tool
def deploy(project: str, region: str) -> str:
    """Deploy a project to a region.

    Args:
        project: Project name.
        region: Cloud region.
    """
    return f"deployed {project} to {region}"


def answer_from_memory(messages: list[dict]) -> str:
    """A scripted model that deploys to whatever region its system prompt mentions."""
    system = messages[0]["content"]
    region = next((w for w in system.split() if w.count("-") == 2 and w[-1].isdigit()), "NO-REGION-IN-CONTEXT")
    return call("deploy", project="atlas", region=region)


def session(name: str, memory: AgentMemory | None, workspace: str) -> None:
    model = ScriptedModel([answer_from_memory, "Done."])
    agent = Agent(model, ToolRegistry([deploy]), workspace, config=AgentConfig(checkpoint_on_start=False))
    if memory is not None:
        attach_memory(agent, memory, budget=400, clock=lambda: 2.0)
    agent.run("Deploy atlas to its usual region.")
    response = agent.log.of_type("ModelResponse")[0].data
    result = agent.log.of_type("ToolResult")[0].data
    print(f"{name}\n  context tokens by section: {response['context']}")
    print(f"  tool call: {response['tool_calls'][0]['arguments']}\n  tool result: {result['content']}")


def main() -> None:
    with tempfile.TemporaryDirectory() as workspace:
        session("no memory (plain C10 agent)", None, workspace)
        memory = AgentMemory()
        memory.observe(MemoryEvent("The deploy region for atlas is eu-west-1.", "user", 1.0))
        memory.observe(MemoryEvent("Heads up: we moved the deploy region for atlas to us-east-2.", "user", 1.5))
        session("with memory (attach_memory, one call)", memory, workspace)
        print("episodes written by the hook:", [e.action for e in memory.episodes])
        print("episodic memory from end_task:", [it.text for it in memory.retriever.items.values() if it.kind == "episodic"])


if __name__ == "__main__":
    main()
