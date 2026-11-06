from agentmem.tokens import count_tokens, truncate_to_tokens
from agentmem.working import PlanStep, TaskState


def test_count_tokens_words_and_punctuation():
    assert count_tokens("deploy atlas.") == 3
    assert count_tokens("") == 0
    assert count_tokens("internationalization") == 3  # long words cost more, as in BPE


def test_truncate_respects_budget_in_every_mode():
    text = "\n".join(f"line {i}: status=200 latency={i}ms" for i in range(200))
    for keep in ("head", "tail", "both"):
        for budget in (5, 20, 64, 300):
            out = truncate_to_tokens(text, budget, keep=keep)
            assert count_tokens(out) <= budget, (keep, budget)
    both = truncate_to_tokens(text, 64, keep="both")
    assert both.startswith("line 0") and both.endswith("line 199: status=200 latency=199ms")
    assert "lines omitted" in both


def test_task_state_apply_render_roundtrip():
    s = TaskState("deploy atlas", plan=[PlanStep("inspect"), PlanStep("fix")])
    s.apply({"done": ["inspect"], "current_step": "fix", "ask": ["which region?"]})
    assert "[x] inspect" in s.render() and "[ ] fix" in s.render()
    s.apply({"answered": ["which region?"], "constraints": ["no deploys on Friday"]})
    assert s.open_questions == []
    assert TaskState.from_json(s.to_json()) == s
