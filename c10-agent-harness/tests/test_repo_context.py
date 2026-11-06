"""Steps 10-13, 15-16: context builder, search, symbols, repo map, plan, compaction, truncation."""

from __future__ import annotations

import pytest
from helpers import make_ctx, toy_workspace

from harness.compaction import SUMMARY_HEADER, extractive_summary, plan_compaction, turn_starts
from harness.context import AgentState, ContextBuilder, repo_section
from harness.messages import Message, ToolCall
from harness.planning import Plan, update_plan
from harness.repomap import build_repo_map, rank_files
from harness.search import (SymbolIndex, find_files, find_symbol, search, search_gitgrep,
                            search_python)
from harness.tokens import count_message_tokens, count_tokens
from harness.tools import ToolError
from harness.truncation import head_only, truncate_output


def test_search_tool_finds_the_definition(tmp_path):
    out = search(make_ctx(toy_workspace(tmp_path)), "def median")
    assert out == "stats_lib/core.py:10: def median(xs: list[float]) -> float:"


def test_search_backends_agree(tmp_path):
    ws = toy_workspace(tmp_path)
    key = lambda ms: sorted((m.path, m.line) for m in ms)  # noqa: E731
    assert key(search_gitgrep(ws, "median")) == key(search_python(ws, "median"))


def test_search_caps_results_and_handles_no_match(tmp_path):
    ctx = make_ctx(toy_workspace(tmp_path))
    assert "more matches" in search(ctx, "def|assert", max_results=2)
    assert "no matches" in search(ctx, "zzzz")


def test_python_backend_rejects_bad_regex(tmp_path):
    with pytest.raises(ToolError, match="invalid regex"):
        search_python(tmp_path, "(")


def test_find_files_glob(tmp_path):
    assert "tests/test_stats.py" in find_files(make_ctx(toy_workspace(tmp_path)), "tests/*")


def test_symbol_index_definitions_and_references(tmp_path):
    ws = toy_workspace(tmp_path)
    idx = SymbolIndex(ws)
    d = idx.definitions("median")[0]
    assert (d.path, d.line, d.kind) == ("stats_lib/core.py", 10, "function")
    assert "tests/test_stats.py" in idx.referencing_files("median")
    assert "used in:" in find_symbol(make_ctx(ws), "median")


def test_symbol_index_survives_syntax_errors(tmp_path):
    (tmp_path / "bad.py").write_text("def f(:\n")
    assert SymbolIndex(tmp_path).files["bad.py"].symbols == []


def test_repo_map_shows_signatures_and_ranks_by_use(tmp_path):
    ws = toy_workspace(tmp_path)
    text = build_repo_map(ws)
    assert "def median(xs: list[float]) -> float" in text
    assert rank_files(SymbolIndex(ws))[0][0] == "stats_lib/core.py"


def test_repo_map_respects_budget_and_says_what_it_dropped(tmp_path):
    ws = toy_workspace(tmp_path)
    small = build_repo_map(ws, max_tokens=45)
    assert count_tokens(small) < 80 and "more files not shown" in small


def test_repo_map_focus_boosts_named_files(tmp_path):
    ws = toy_workspace(tmp_path)
    assert rank_files(SymbolIndex(ws), focus="summarize the report")[0][0] == "stats_lib/report.py"


def test_plan_tool_round_trip():
    class Ctx:
        plan = Plan()
    out = update_plan(Ctx, ["[x] inspect", "[>] fix", "[ ] test"])
    assert "[x] inspect\n[>] fix\n[ ] test" in out and Ctx.plan.current.title == "fix"
    with pytest.raises(ToolError, match="at most one"):
        update_plan(Ctx, ["[>] a", "[>] b"])
    with pytest.raises(ToolError, match="must start with"):
        update_plan(Ctx, ["no mark"])


def test_context_builder_sections_and_breakdown(tmp_path):
    ws = toy_workspace(tmp_path)
    state = AgentState("fix median", ws)
    builder = ContextBuilder(sections={"repo": repo_section()})
    builder.add_section("memory", lambda s: "## Memory\nremember this")
    ctx = builder.build(state, [Message("user", "fix median")], [{"type": "function"}])
    system = ctx.messages[0]["content"]
    assert ctx.messages[0]["role"] == "system" and ctx.messages[1]["role"] == "user"
    for needle in ("## Repository map", "## Git status", "## Memory", "## Task\nfix median", "(no plan yet)"):
        assert needle in system
    assert set(ctx.breakdown) == {"instructions", "repo", "memory", "task", "tools", "history"}
    assert ctx.tokens == sum(ctx.breakdown.values())


def test_section_returning_none_is_skipped(tmp_path):
    builder = ContextBuilder(sections={"empty": lambda s: None})
    assert "empty" not in builder.build(AgentState("t", tmp_path), [], []).breakdown


def test_token_estimate_counts_tool_arguments():
    plain = [{"role": "assistant", "content": "x"}]
    with_call = [{"role": "assistant", "content": "x", "tool_calls": [
        {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]}]
    assert count_message_tokens(with_call) > count_message_tokens(plain) > 0


def _turn(i: int, content: str = "ok") -> list[Message]:
    call = ToolCall("read_file", {"path": f"f{i}.py"}, id=f"c{i}")
    return [Message("assistant", f"turn {i}", [call]),
            Message("tool", content, tool_call_id=f"c{i}", name="read_file")]


def test_compaction_keeps_recent_turns_whole_and_pairs_intact():
    history = [Message("user", "task")] + [m for i in range(8) for m in _turn(i)]
    plan = plan_compaction(history, keep_recent_turns=3)
    kept = history[plan.cut:]
    assert kept[0].role == "assistant" and len(turn_starts(kept)) == 3
    called = {c.id for m in kept for c in m.tool_calls}
    assert {m.tool_call_id for m in kept if m.role == "tool"} == called


def test_compaction_returns_none_when_nothing_is_old_enough():
    assert plan_compaction(_turn(0), keep_recent_turns=4) is None


def test_summary_keeps_failures_verbatim():
    history = _turn(0, "tests: FAIL (1 failed)\n  a.py:3 [test_a] assert 1 == 2") + _turn(1, "fine")
    text = extractive_summary(history)
    assert text.startswith(SUMMARY_HEADER)
    assert "read_file(path='f1.py') -> fine" in text
    assert "Important failures" in text and "assert 1 == 2" in text


def test_second_compaction_extends_the_first():
    first = extractive_summary(_turn(0, "Error: boom"))
    second = extractive_summary([Message("user", first)] + _turn(1, "fine"))
    assert second.count(SUMMARY_HEADER) == 1
    assert "f0.py" in second and "f1.py" in second and "Error: boom" in second
    assert second.count("Important failures") == 1


def test_truncation_passthrough_when_small():
    t = truncate_output("a\nb\n", max_lines=10)
    assert not t.truncated and t.text == "a\nb"


def test_truncation_keeps_head_tail_and_rescues_errors():
    lines = [f"noise {i}" for i in range(5000)]
    lines[2500] = "E   AssertionError: assert 3 == 2.5"
    text = "\n".join(lines)
    t = truncate_output(text, max_lines=60)
    assert t.truncated and t.total_lines == 5000 and t.shown_lines <= 60
    assert "noise 0\n" in t.text and "noise 4999" in t.text and "assert 3 == 2.5" in t.text
    assert "lines omitted" in t.text
    assert "assert 3 == 2.5" not in head_only(text, 60)


def test_truncation_clips_long_lines_and_spills(tmp_path):
    spill = tmp_path / "out.txt"
    text = "\n".join(["x" * 5000] + ["y"] * 500)
    t = truncate_output(text, max_lines=20, spill_path=spill, spill_name="out.txt")
    assert len(t.text.splitlines()[0]) < 450 and spill.read_text() == text and "out.txt" in t.text
