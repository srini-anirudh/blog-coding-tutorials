"""AgentMemory inside the real C10 harness. Skipped only if ../c10-agent-harness is absent."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

C10 = Path(__file__).resolve().parents[2] / "c10-agent-harness"
if not C10.exists():
    pytest.skip("c10-agent-harness not present", allow_module_level=True)

from agentmem import AgentMemory, BudgetError, MemoryEvent, TaskState  # noqa: E402
from agentmem.integration import BudgetedContextBuilder, attach_memory, group_turns  # noqa: E402
from agentmem.bench.deploy_eval import CONDITIONS, run_deploy  # noqa: E402
from agentmem.bench.deploy_harness import run_deploy_harness  # noqa: E402
from agentmem.bench.deploy_world import make_world  # noqa: E402
import harness  # noqa: E402
from harness import (Agent, AgentConfig, AgentState, ContextBuilder, Message, OpenAICompatModel,  # noqa: E402
                     ScriptedModel, ToolCall, ToolOutput, ToolRegistry, call, coding_tools, tool)


@tool
def echo(text: str) -> str:
    """Echo text.

    Args:
        text: What to echo.
    """
    return text


def plain_agent(model, workspace, tools=(echo,), **config):
    return Agent(model, ToolRegistry(list(tools)), workspace, config=AgentConfig(checkpoint_on_start=False, **config))


def test_the_harness_under_test_is_c10_and_has_the_surface_we_use(tmp_path):
    """Everything agentmem relies on from C10. If C10 changes one, this names it."""
    assert Path(harness.__file__).resolve().parents[1] == C10.resolve()
    names = ["Agent", "AgentConfig", "AgentState", "ContextBuilder", "Event", "EventLog", "HFLocalModel", "Message",
             "ModelResponse", "OpenAICompatModel", "PermissionPolicy", "Risk", "ScriptedModel", "ToolCall",
             "ToolOutput", "ToolRegistry", "call", "coding_tools", "tool"]
    assert [n for n in names if n not in harness.__all__] == []
    agent = plain_agent(ScriptedModel(["done"]), tmp_path)
    assert [a for a in ("builder", "log", "state", "tools", "run") if not hasattr(agent, a)] == []
    assert [a for a in ("subscribe", "of_type", "events") if not hasattr(agent.log, a)] == []
    assert [a for a in ("instructions", "sections", "build") if not hasattr(agent.builder, a)] == []
    assert [a for a in ("task", "plan", "step", "notes") if not hasattr(agent.state, a)] == []
    agent.run("hello")
    assert [e.type for e in agent.log.events] == ["UserMessage", "ModelResponse", "Final"]
    assert agent.log.events[0].step == 0 and agent.log.events[0].data["content"] == "hello"


def test_memory_section_reaches_the_model_and_events_reach_memory(tmp_path):
    mem = AgentMemory()
    mem.observe(MemoryEvent("The deploy region for atlas is eu-west-1.", "user", 0.5))
    model = ScriptedModel([call("echo", text="hi"), "done"])
    agent = plain_agent(model, tmp_path)
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


def test_attach_memory_keeps_the_sections_of_the_agents_builder(tmp_path):
    model = ScriptedModel(["done"])
    builder = ContextBuilder(sections={"repo": lambda state: "## Repository map\nstats_lib/core.py"})
    agent = Agent(model, ToolRegistry([echo]), tmp_path, context=builder, config=AgentConfig(checkpoint_on_start=False))
    attach_memory(agent, AgentMemory(), budget=800)
    agent.run("go")
    assert "stats_lib/core.py" in model.seen[0][0]["content"]
    assert agent.log.of_type("ModelResponse")[0].data["context"]["repo"] > 0


def test_session_notes_and_working_memory_reach_the_prompt(tmp_path):
    mem = AgentMemory()
    mem.working = TaskState("ship the audit fix", current_step="run the tests")
    mem.observe(MemoryEvent("Just for today, the owner for atlas is marco.", "user", 0.5))
    model = ScriptedModel(["done"])
    agent = plain_agent(model, tmp_path)
    attach_memory(agent, mem, budget=800)
    agent.run("Who owns atlas?")
    system = model.seen[0][0]["content"]
    assert "today only: Just for today, the owner for atlas is marco." in system
    assert "goal: ship the audit fix" in system and "current step: run the tests" in system


def test_a_run_that_hits_max_steps_is_still_remembered(tmp_path):
    mem = AgentMemory()
    agent = plain_agent(ScriptedModel([call("echo", text="a"), call("echo", text="b")]), tmp_path, max_steps=2)
    attach_memory(agent, mem, budget=400)
    assert agent.run("never finishes").status == "max_steps"
    assert [it.text for it in mem.retriever.items.values()] == ["episode task1 (session 1): `echo a` -> ok; `echo b` -> ok"]


def test_each_run_of_one_agent_is_its_own_episode(tmp_path):
    mem = AgentMemory()
    agent = plain_agent(ScriptedModel([call("echo", text="a"), "x", call("echo", text="b"), "y"]), tmp_path)
    attach_memory(agent, mem, budget=400)
    agent.run("first")
    agent.run("second")
    assert sorted(it.text for it in mem.retriever.items.values()) == [
        "episode task1 (session 1): `echo a` -> ok", "episode task2 (session 1): `echo b` -> ok"]


def test_default_hook_gives_consolidation_what_it_needs(tmp_path):
    """Fail, fix, retry through the real loop, with no custom reward: a rule must come out."""
    logged_in = set()

    @tool
    def deploy(project: str) -> ToolOutput:
        """Deploy a project.

        Args:
            project: The project name.
        """
        return ToolOutput("healthy", False) if project in logged_in else ToolOutput("AuthError: rejected", True)

    @tool
    def login(project: str) -> str:
        """Log in to a project.

        Args:
            project: The project name.
        """
        logged_in.add(project)
        return "ok"

    mem = AgentMemory()
    for project in ("atlas", "cobalt"):
        script = [call("deploy", project=project), call("login", project=project), call("deploy", project=project), "done"]
        agent = plain_agent(ScriptedModel(script), tmp_path, tools=(deploy, login))
        attach_memory(agent, mem, budget=800)
        agent.run(f"Deploy {project}.")
    assert [p.steps for p in mem.end_session()] == [["login {entity}"]]


def test_a_budget_below_the_tool_schemas_fails_when_attaching(tmp_path):
    agent = Agent(ScriptedModel(["done"]), coding_tools(), tmp_path, config=AgentConfig(checkpoint_on_start=False))
    with pytest.raises(BudgetError, match="tool schemas alone cost"):
        attach_memory(agent, AgentMemory(), budget=400)


def test_a_long_opening_request_leaves_room_for_the_newest_turn(tmp_path):
    history = [Message("user", "please read this " * 400),
               Message("assistant", "", [ToolCall("echo", {"text": "word " * 400}, id="c1")]),
               Message("tool", "NEWEST-RESULT\n" + "line\n" * 400, tool_call_id="c1", name="echo")]
    tools = ToolRegistry([echo]).schemas()
    ctx = BudgetedContextBuilder(600, instructions="Be brief.").build(AgentState("go", tmp_path), history, tools)
    assert ctx.tokens <= 600 + 8
    roles = [m["role"] for m in ctx.messages]
    assert roles == ["system", "user", "assistant", "tool"]
    assert "NEWEST-RESULT" in ctx.messages[-1]["content"]
    assert isinstance(ctx.messages[2]["tool_calls"][0]["function"]["arguments"]["text"], str)  # still a well-formed call
    assert history[1].tool_calls[0].arguments["text"] == "word " * 400                         # the history is untouched


def test_harness_compaction_and_the_budgeted_builder_work_together(tmp_path):
    script = [call("echo", text="x " * 300) for _ in range(8)] + ["done"]
    agent = plain_agent(ScriptedModel(script), tmp_path, context_budget=600, keep_recent_turns=1)
    attach_memory(agent, AgentMemory(), budget=800)
    assert agent.run("go").status == "done"
    assert agent.log.of_type("Compaction")


def test_memory_works_over_the_openai_client_against_a_local_stub(tmp_path, monkeypatch):
    bodies = []
    replies = [{"role": "assistant", "content": None, "tool_calls": [
                   {"id": "c1", "type": "function", "function": {"name": "echo", "arguments": json.dumps({"text": "hi"})}}]},
               {"role": "assistant", "content": "done"}]

    class Stub(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps({"choices": [{"message": replies[len(bodies) - 1]}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    monkeypatch.setenv("HARNESS_MODEL", "stub")
    mem = AgentMemory()
    mem.observe(MemoryEvent("The deploy region for atlas is eu-west-1.", "user", 0.5))
    agent = plain_agent(OpenAICompatModel(), tmp_path)
    attach_memory(agent, mem, budget=400)
    try:
        assert agent.run("What is the deploy region for atlas?").status == "done"
    finally:
        server.shutdown()
    assert "atlas deploy region: eu-west-1" in bodies[0]["messages"][0]["content"]
    sent = bodies[1]["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(sent) == {"text": "hi"}                # the wire format wants a JSON string
    assert len(mem.episodes) == 1


@pytest.mark.slow
@pytest.mark.network
def test_real_model_through_the_real_agent_across_two_processes(tmp_path):
    """Two projects with the real model. It checks the plumbing, not how clever the model is."""
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_real_agent.py"
    subprocess.run([sys.executable, str(script), "--projects", "2", "--out", str(tmp_path)], check=True)
    out = json.loads((tmp_path / "real_agent.json").read_text())
    assert out["separate_processes"] and out["harness"] == "../c10-agent-harness/harness"
    assert out["summary"]["retrieved"]["fact_in_prompt"] == 2 and out["summary"]["no_memory"]["fact_in_prompt"] == 0
