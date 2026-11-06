from harness_lab.code_action import code_action_agent, json_tool_agent, make_repo, snapshot


def test_both_agents_make_the_same_change(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    hits = make_repo(a, 12)
    make_repo(b, 12)
    tj, tc = json_tool_agent(a, "rename"), code_action_agent(b, "rename")
    assert snapshot(a) == snapshot(b)
    assert not any("parse_cfg" in s for s in snapshot(a).values())
    assert tj.model_calls == 1 + 12 + hits + 1
    assert tc.model_calls == 2
    assert tj.input_tokens > 10 * tc.input_tokens
