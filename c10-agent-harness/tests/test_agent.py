"""Steps 1 and 14, 18, plus the whole loop: replay, retries, failure paths, the demo."""

from __future__ import annotations

import json

import pytest
from helpers import toy_workspace

from harness import (Agent, AgentConfig, EventLog, PermissionPolicy, Risk, ScriptedModel, ToolRegistry,
                     Tracer, call, coding_tools, tool)
from harness.loop_minimal import run_minimal
from harness.messages import reset_call_ids
from harness.models import ModelError, ScriptExhausted, TransientModelError


@tool
def add(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a + b)


def fast_config(**kw) -> AgentConfig:
    return AgentConfig(retry_backoff_s=0.0, checkpoint_on_start=False, **kw)


def test_minimal_loop_runs_tool_then_stops():
    model = ScriptedModel([call("add", a=2, b=3), "The answer is 5."])
    ctx = run_minimal(model, {"add": lambda a, b: a + b}, "add 2 and 3")
    assert [m["role"] for m in ctx] == ["user", "assistant", "tool", "assistant"]
    assert ctx[2]["content"] == "5" and ctx[-1]["content"] == "The answer is 5."


def test_minimal_loop_crashes_on_unknown_tool():
    with pytest.raises(KeyError):
        run_minimal(ScriptedModel([call("nope")]), {}, "t")


def test_minimal_loop_spins_on_malformed_call():
    # The malformed call has no tool_calls and is not final, so the naive loop just calls again.
    model = ScriptedModel(['<tool_call>{bad</tool_call>'] * 3)
    with pytest.raises(ScriptExhausted):
        run_minimal(model, {}, "t")
    assert model.calls == 3


def _agent(tmp_path, script, tools=None, **kw):
    config = kw.pop("config", fast_config())
    return Agent(ScriptedModel(script), tools or ToolRegistry([add]), tmp_path, config=config, **kw)


def test_agent_happy_path(tmp_path):
    a = _agent(tmp_path, [call("add", a=1, b=2), "three"])
    r = a.run("add")
    assert (r.status, r.final_text, r.steps) == ("done", "three", 2)
    assert a.log.events[-1].type == "Final"
    assert [e.type for e in a.log.events][:4] == ["UserMessage", "ModelResponse", "ToolCall", "ToolResult"]


def test_validation_errors_go_back_and_the_model_recovers(tmp_path):
    a = _agent(tmp_path, [call("add", a="one", b=2), call("nope"), call("add", a=1, b=2), "ok"])
    assert a.run("x").status == "done"
    results = [e.data for e in a.log.of_type("ToolResult")]
    assert results[0]["is_error"] and "expected integer" in results[0]["content"]
    assert "unknown tool" in results[1]["content"] and not results[2]["is_error"]


def test_malformed_call_gets_feedback_then_recovers(tmp_path):
    a = _agent(tmp_path, ['<tool_call>{"name": "add", "arguments": {"a": 1,}}</tool_call>', "fine"])
    assert a.run("x").status == "done"
    feedback = [e for e in a.log.of_type("UserMessage") if e.data.get("kind") == "feedback"]
    assert len(feedback) == 1 and "could not be parsed" in feedback[0].data["content"]


def test_too_many_protocol_errors_stops_the_run(tmp_path):
    a = _agent(tmp_path, ["<tool_call>{bad</tool_call>"] * 10, config=fast_config(max_protocol_errors=3))
    r = a.run("x")
    assert r.status == "protocol_errors" and r.steps == 3


def test_step_budget_stops_the_run(tmp_path):
    a = _agent(tmp_path, [call("add", a=1, b=1)] * 10, config=fast_config(max_steps=4))
    assert a.run("x").status == "max_steps"


def test_transient_errors_retry_and_are_traced(tmp_path):
    a = _agent(tmp_path, [TransientModelError("429"), TransientModelError("503"), "ok"])
    r = a.run("x")
    assert r.status == "done" and r.trace["retries"] == 2
    assert len([e for e in a.log.of_type("Error") if e.data["kind"] == "transient"]) == 2


def test_permanent_model_error_ends_the_run(tmp_path):
    a = _agent(tmp_path, [ModelError("401 bad key")])
    r = a.run("x")
    assert r.status == "model_error" and a.log.of_type("Error")[0].data["kind"] == "model"


def test_denied_call_never_executes_and_is_logged(tmp_path):
    ran = []

    @tool(risk=Risk.EXTERNAL)
    def deploy() -> str:
        """Deploy."""
        ran.append(1)
        return "deployed"

    asked = []
    policy = PermissionPolicy(approver=lambda n, a, r: asked.append((n, r)) or False)
    a = _agent(tmp_path, [call("deploy"), "gave up"], tools=ToolRegistry([deploy]), policy=policy)
    assert a.run("x").status == "done"
    assert ran == [] and asked == [("deploy", Risk.EXTERNAL)]
    ap = a.log.of_type("Approval")[0].data
    assert not ap["allowed"] and ap["asked"] and ap["risk"] == "EXTERNAL"
    assert "Permission denied" in a.log.of_type("ToolResult")[0].data["content"]


def test_tool_bug_does_not_kill_the_loop(tmp_path):
    @tool
    def buggy() -> str:
        """Crash."""
        raise ZeroDivisionError("oops")

    a = _agent(tmp_path, [call("buggy"), "noted"], tools=ToolRegistry([buggy]))
    assert a.run("x").status == "done"
    assert "ZeroDivisionError" in a.log.of_type("ToolResult")[0].data["content"]


def test_giant_observation_is_truncated_by_the_loop(tmp_path):
    @tool
    def noisy() -> str:
        """Print a lot."""
        return "\n".join(f"line {i}" for i in range(10_000))

    a = _agent(tmp_path, [call("noisy"), "ok"], tools=ToolRegistry([noisy]))
    a.run("x")
    assert len(a.log.of_type("ToolResult")[0].data["content"].splitlines()) < 170


def test_compaction_triggers_preserves_task_and_replays(tmp_path):
    @tool
    def blob(i: int) -> str:
        """Return bulky text.

        Args:
            i: Index.
        """
        return f"blob {i} " + "word " * 200

    script = [call("blob", i=i) for i in range(12)] + ["done"]
    a = _agent(tmp_path, script, tools=ToolRegistry([blob]),
               config=fast_config(context_budget=800, keep_recent_turns=2))
    assert a.run("find the blob").status == "done"
    comp = a.log.of_type("Compaction")
    assert comp and comp[0].data["tokens_after"] < comp[0].data["tokens_before"]
    # The task stays visible to the model even though its history message was compacted away.
    last_seen = a.model.seen[-1]
    assert "## Task\nfind the blob" in last_seen[0]["content"]
    assert "Summary of earlier turns" in last_seen[1]["content"]
    assert a.log.to_messages() == [m for _, m in a.history]


def test_replay_survives_consecutive_compactions(tmp_path):
    # Regression: the second cut can land before the first summary's seq number.
    @tool
    def blob(i: int) -> str:
        """Return bulky text.

        Args:
            i: Index.
        """
        return "word " * 150

    script = [call("blob", i=i) for i in range(10)] + ["done"]
    a = _agent(tmp_path, script, tools=ToolRegistry([blob]),
               config=fast_config(context_budget=600, keep_recent_turns=4))
    assert a.run("x").status == "done" and len(a.log.of_type("Compaction")) >= 2
    assert a.log.to_messages() == [m for _, m in a.history]


def test_compaction_does_not_refire_every_step(tmp_path):
    @tool
    def blob(i: int) -> str:
        """Return bulky text.

        Args:
            i: Index.
        """
        return "word " * 120

    script = [call("blob", i=i) for i in range(16)] + ["done"]
    a = _agent(tmp_path, script, tools=ToolRegistry([blob]),
               config=fast_config(context_budget=1200, keep_recent_turns=3))
    a.run("x")
    assert 0 < len(a.log.of_type("Compaction")) <= 8


def test_event_log_jsonl_round_trip(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    a = _agent(tmp_path, [call("add", a=1, b=2), "ok"], log=EventLog(path))
    a.run("x")
    lines = path.read_text().splitlines()
    assert len(lines) == len(a.log.events) and all(json.loads(x)["type"] for x in lines)
    assert [json.loads(x)["seq"] for x in lines] == list(range(len(lines)))
    assert EventLog.load(path).to_messages() == [m for _, m in a.history]


def test_event_log_rejects_unknown_types_and_notifies_subscribers():
    log = EventLog()
    seen = []
    log.subscribe(seen.append)
    log.append("Final", 1, text="x")
    assert seen[0].type == "Final"
    with pytest.raises(ValueError):
        log.append("Nonsense", 1)


def test_tracer_summary_counts_tokens_and_failures(tmp_path):
    tr = Tracer()
    a = _agent(tmp_path, [call("add", a=1, b=2), call("add", a=1, b=2), "ok"], tracer=tr)
    r = a.run("x")
    assert r.trace["model_calls"] == 3 and r.trace["tool_calls"] == 2
    assert r.trace["prompt_tokens"] > 0 and r.trace["by_tool"]["add"]["calls"] == 2
    assert "model calls" in tr.table()


def test_end_to_end_fixes_the_failing_test(tmp_path):
    """The demo's loop on a toy repo: real tools, real pytest, real edit, scripted decisions."""
    reset_call_ids()
    ws = toy_workspace(tmp_path)
    old = "    mid = len(s) // 2\n    return s[mid]"
    new = "    mid = len(s) // 2\n    if len(s) % 2 == 0:\n        return (s[mid - 1] + s[mid]) / 2\n    return s[mid]"
    script = [call("run_tests"), call("read_file", path="stats_lib/core.py"),
              call("edit_file", path="stats_lib/core.py", search=old, replace=new),
              call("run_tests"), "Fixed."]
    a = Agent(ScriptedModel(script), coding_tools(), ws, config=AgentConfig())
    r = a.run("fix median")
    assert r.status == "done"
    tests = [e.data for e in a.log.of_type("TestResult")]
    assert (tests[0]["status"], tests[0]["failed"]) == ("fail", 2)
    assert (tests[1]["status"], tests[1]["passed"]) == ("pass", 6)
    assert a.log.of_type("Patch")[0].data["format"] == "search_replace"
    assert a.log.of_type("Checkpoint")


def test_full_registry_exposes_twenty_valid_schemas():
    reg = coding_tools()
    assert len(reg.names) == 20
    for s in reg.schemas():
        assert s["function"]["description"] and s["function"]["parameters"]["type"] == "object"
