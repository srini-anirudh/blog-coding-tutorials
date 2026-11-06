from harness_lab.observation import apply_policy, build_output, grep_output, head_and_tail, pytest_output


def test_head_and_tail_keeps_both_ends():
    text = "\n".join(f"line {i}" for i in range(5000))
    out = head_and_tail(text, 400)
    assert "line 0" in out and "line 4999" in out and "line 2500" not in out


def test_build_error_in_the_middle(tmp_path):
    text, key = build_output(3000, 0.5)
    assert key in text
    assert key not in apply_policy("head and tail", text, "build", 500, "error")
    assert key in apply_policy("summary", text, "build", 500, "error")


def test_pytest_failure_survives_summary(tmp_path):
    text, key = pytest_output(tmp_path, 40, 20)
    assert key in text
    assert key in apply_policy("summary", text, "pytest", 800, r"^E\s")


def test_grep_answer_needs_a_follow_up(tmp_path):
    text, key = grep_output(tmp_path, 200, 0.5)
    assert key not in apply_policy("summary", text, "grep", 800, "security")
    assert key in apply_policy("spill to handle", text, "grep", 800, "security")
