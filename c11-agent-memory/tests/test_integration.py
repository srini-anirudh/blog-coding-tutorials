"""AgentMemory inside the real C10 harness. Skipped only if ../c10-agent-harness is absent."""

from pathlib import Path

import pytest

C10 = Path(__file__).resolve().parents[2] / "c10-agent-harness"
if not C10.exists():
    pytest.skip("c10-agent-harness not present", allow_module_level=True)

from agentmem import AgentMemory, count_tokens  # noqa: E402
from agentmem.integration import BudgetedContextBuilder, attach_memory, group_turns  # noqa: E402
from agentmem.bench.deploy_eval import CONDITIONS, run_deploy  # noqa: E402
from agentmem.bench.deploy_harness import run_deploy_harness  # noqa: E402
from agentmem.bench.deploy_world import make_world  # noqa: E402
from harness import Agent, AgentConfig, AgentState, Message, ScriptedModel, ToolCall, ToolRegistry, call, tool  # noqa: E402
from harness.tokens import count_message_tokens  # noqa: E402


@tool
def echo(text: str) -> str:
    """Echo text.

    Args:
        text: What to echo.
    """
    return text


def test_memory_section_reaches_the_model_and_events_reach_memory(tmp_path):
    mem = AgentMemory()
    mem.observe(__import__("agentmem").MemoryEvent("The deploy region for atlas is eu-west-1.", "user", 0.5))
    model = ScriptedModel([call("echo", text="hi"), "done"])
    agent = Agent(model, ToolRegistry([echo]), tmp_path, config=AgentConfig(checkpoint_on_start=False))
    attach_memory(agent, mem, budget=400)
    result = agent.run("What is the deploy region for atlas?")
    assert result.status == "done"
    system = model.seen[0][0]["content"]
    assert "atlas deploy region: eu-west-1" in system     # READ: the memory is in the prompt
    assert len(mem.episodes) == 1                         # WRITE: the tool call is an episode
    assert mem.episodes.episodes[0].action == "echo hi"
    assert any("echo hi" in it.text for it in mem.retriever.items.values())  # end_task summarised it


def test_group_turns_keeps_calls_with_results():
    h = [Message("user", "go"), Message("assistant", "", [ToolCall("a", {}, id="1"), ToolCall("b", {}, id="2")]),
         Message("tool", "x", tool_call_id="1", name="a"), Message("tool", "y", tool_call_id="2", name="b"),
         Message("assistant", "done")]
    assert [len(t) for t in group_turns(h)] == [1, 3, 1]


def test_budgeted_builder_respects_budget_and_never_splits_a_call(tmp_path):
    history = [Message("user", "first request")]
    for i in range(30):
        history.append(Message("assistant", "", [ToolCall("echo", {"text": "x"}, id=f"c{i}")]))
        history.append(Message("tool", "\n".join(f"log {j} of step {i}" for j in range(40)),
                               tool_call_id=f"c{i}", name="echo"))
    builder = BudgetedContextBuilder(600, instructions="Be brief.")
    ctx = builder.build(AgentState("first request", tmp_path), history, ToolRegistry([echo]).schemas())
    assert ctx.tokens <= 600 + 8  # the task section is rendered, the framing estimate is per message
    roles = [m["role"] for m in ctx.messages]
    assert roles[1] == "user" and roles[-1] == "tool"     # first request and newest result survive
    ids = {c["id"] for m in ctx.messages for c in m.get("tool_calls", [])}
    assert ids == {m["tool_call_id"] for m in ctx.messages if m["role"] == "tool"}


@pytest.mark.parametrize("condition", ["no_memory", "oracle"])
def test_harness_loop_agrees_with_standalone_loop_when_memory_is_not_in_play(condition):
    a = run_deploy(make_world(3), condition, 4096)
    b = run_deploy_harness(make_world(3), condition, 4096)
    assert [(r.steps, r.success) for r in a] == [(r.steps, r.success) for r in b]


def test_memory_saves_steps_through_the_real_loop():
    plain = run_deploy_harness(make_world(1), "no_memory", 1024)
    remembered = run_deploy_harness(make_world(1), "retrieved", 1024)
    assert sum(r.steps for r in remembered) < sum(r.steps for r in plain)
    assert sum(r.success for r in remembered) >= sum(r.success for r in plain)
