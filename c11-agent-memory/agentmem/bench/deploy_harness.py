"""The deploy world through the real C10 Agent loop.

`deploy_eval.run_deploy` is a small standalone loop, so this folder works without C10.
This module runs the same policy on the same world through `harness.Agent`, with real
tool registration, validation, permissions and the JSONL-ready event log. The two
must agree on behaviour, which `tests/test_integration.py` checks.
"""

from __future__ import annotations

import re
import tempfile

from ..consolidate import error_signature
from ..integration import BudgetedContextBuilder, MemoryHook, ReaderModel  # puts ../c10-agent-harness on sys.path
from ..memory import AgentMemory
from .deploy_eval import INSTRUCTIONS, DeployPolicy, TaskResult, oracle_line
from .deploy_world import DeployWorld

from harness import Agent, AgentConfig, ModelResponse, PermissionPolicy, Risk, ToolCall, ToolOutput, ToolRegistry, tool  # noqa: E402,I001

_TASK = re.compile(r"\((?P<tid>s\d+t\d+)\) Deploy (?P<project>\w+)\.")


def deploy_tools(world: DeployWorld) -> ToolRegistry:
    """One @tool per world action. Each returns the observation; a failed deploy is an error."""
    def make(name: str):
        def run(project: str) -> ToolOutput:
            obs, reward = world.step(f"{name} {project}")
            return ToolOutput(obs, is_error=reward < 0)
        run.__name__ = name
        run.__doc__ = f"Run {name} on a project.\n\nArgs:\n    project: The project name.\n"
        return tool(run, name=name, risk=Risk.READ)
    return ToolRegistry([make(n) for n in ("deploy", "migrate", "login", "build_assets", "clear_cache")])


def render_for_policy(messages: list[dict]) -> str:
    """The chat as the text format DeployPolicy reads: system text, then one line per step."""
    lines = [messages[0]["content"]]
    tid = ""
    calls: dict[str, str] = {}
    for m in messages[1:]:
        if m["role"] == "user" and (found := _TASK.search(m["content"])):
            tid = found["tid"]
        for c in m.get("tool_calls", []):
            calls[c["id"]] = f"{c['function']['name']} {c['function']['arguments']['project']}"
        if m["role"] == "tool":
            lines.append(f"[{tid}] `{calls[m['tool_call_id']]}` -> {error_signature(m['content']) or 'ok'}")
    return "\n".join(lines)


def policy_model(policy: DeployPolicy, task: dict[str, str], max_calls: int) -> ReaderModel:
    calls = {"n": 0}

    def respond(messages: list[dict], tools: list[dict]) -> ModelResponse:
        last = messages[-1]
        if last["role"] == "tool" and last["name"] == "deploy" and error_signature(last["content"]) is None:
            return ModelResponse(text=f"Deployed {task['project']}.")
        if calls["n"] == max_calls:
            return ModelResponse(text=f"Giving up on {task['project']} after {max_calls} tool calls.")
        calls["n"] += 1
        action = policy(render_for_policy(messages), task["id"], task["project"])
        name, project = action.split(" ")
        return ModelResponse(tool_calls=[ToolCall(name, {"project": project})])
    return ReaderModel(respond)


def run_deploy_harness(world: DeployWorld, condition: str, budget: int, max_steps: int = 8,
                       policy: DeployPolicy | None = None, memory: AgentMemory | None = None) -> list[TaskResult]:
    """Conditions: no_memory, retrieved, oracle. One fresh Agent per task; memory persists."""
    policy = policy or DeployPolicy()
    memory = memory or AgentMemory()
    tools = deploy_tools(world)
    current = {"id": "", "project": ""}
    results, seen, session = [], set(), 0
    with tempfile.TemporaryDirectory() as workspace:
        for task in world.tasks:
            if task.session != session:
                world.new_session()
                memory.end_session()
                session = task.session
            current.update(id=task.id, project=task.project)
            clock = lambda: task.session + (len(memory.episodes) + 1) / 1000  # noqa: B023
            use_memory = memory if condition == "retrieved" else None
            builder = BudgetedContextBuilder(budget, use_memory, INSTRUCTIONS, clock)
            if condition == "oracle":
                builder.instructions = f"{INSTRUCTIONS}\n{oracle_line(world, task.project)}"
            model = policy_model(policy, current, max_steps)
            agent = Agent(model, tools, workspace, context=builder, policy=PermissionPolicy(),
                          config=AgentConfig(max_steps=max_steps + 1, context_budget=10 * budget,
                                             checkpoint_on_start=False))
            hook = MemoryHook(memory, agent.state, lambda: task.id, lambda: task.session, clock,  # noqa: B023
                              reward_of=_reward)
            agent.log.subscribe(hook)
            agent.run(f"({task.id}) Deploy {task.project}.")
            tool_results = agent.log.of_type("ToolResult")
            success = any(e.data["name"] == "deploy" and not e.data["is_error"] for e in tool_results)
            tokens = sum(model.prompts) / max(1, len(model.prompts))
            results.append(TaskResult(task, condition, success, len(tool_results), tokens, task.project in seen))
            seen.add(task.project)
    return results


def _reward(name: str, args: dict, content: str, is_error: bool) -> float:
    if is_error:
        return -1.0
    return 1.0 if name == "deploy" else 0.0
