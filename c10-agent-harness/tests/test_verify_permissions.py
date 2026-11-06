"""Steps 7 and 9: normalised verification results and the permission layer."""

from __future__ import annotations

import pytest
from helpers import make_ctx, toy_workspace

from harness.permissions import PermissionPolicy, Risk, classify_command
from harness.sandbox import ExecResult
from harness.verify import CheckResult, build, builtin_lint, parse_pytest, run_linter, run_tests, run_typecheck


def test_run_tests_failure_is_normalised(tmp_path):
    out = run_tests(make_ctx(toy_workspace(tmp_path)))
    d = out.data
    assert out.kind == "test" and out.is_error
    assert (d["status"], d["passed"], d["failed"]) == ("fail", 4, 2)
    tests = {x["code"]: x for x in d["diagnostics"]}
    assert tests["test_median_even"]["line"] == 15 and tests["test_summarize"]["line"] == 28
    assert "assert 3 == 2.5" in tests["test_median_even"]["message"]


def test_run_tests_target_narrows_the_run(tmp_path):
    out = run_tests(make_ctx(toy_workspace(tmp_path)), target="tests/test_stats.py::test_mean")
    assert out.data["status"] == "pass" and out.data["passed"] == 1 and not out.is_error


def _pytest_result(stdout: str, code: int, timed_out: bool = False) -> ExecResult:
    return ExecResult("pytest", code, stdout, "", 0.1, timed_out=timed_out)


def test_parse_pytest_counts_and_statuses():
    r = parse_pytest(_pytest_result("3 passed, 1 skipped in 0.1s", 0))
    assert (r.status, r.passed, r.skipped) == ("pass", 3, 1)
    assert parse_pytest(_pytest_result("no tests ran in 0.01s", 5)).status == "error"
    assert parse_pytest(_pytest_result("", None, timed_out=True)).status == "timeout"


def test_builtin_lint_finds_syntax_errors_and_unused_imports(tmp_path):
    (tmp_path / "bad.py").write_text("def f(:\n")
    (tmp_path / "unused.py").write_text("import os\nimport sys\nprint(sys.argv)\n")
    (tmp_path / "skip_init").mkdir()
    (tmp_path / "skip_init" / "__init__.py").write_text("import os\n")
    diags = {(d.file, d.code): d for d in builtin_lint(tmp_path)}
    assert ("bad.py", "E999") in diags and ("unused.py", "F401") in diags
    assert "'os'" in diags[("unused.py", "F401")].message
    assert not any("__init__" in f for f, _ in diags)


def test_run_linter_tool_returns_a_check_result(tmp_path):
    (tmp_path / "unused.py").write_text("import os\n")
    out = run_linter(make_ctx(tmp_path))
    assert out.data["status"] == "fail" and out.data["diagnostics"][0]["code"] == "F401"


def test_typecheck_reports_unavailable_or_a_result(tmp_path):
    (tmp_path / "a.py").write_text("x: int = 1\n")
    assert run_typecheck(make_ctx(tmp_path)).data["status"] in ("unavailable", "pass")


def test_build_reports_compile_errors(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n")
    assert build(make_ctx(tmp_path)).data["status"] == "pass"
    (tmp_path / "bad.py").write_text("def f(:\n")
    out = build(make_ctx(tmp_path))
    assert out.data["status"] == "fail" and out.data["diagnostics"][0]["line"] == 1


def test_check_result_render_is_one_shape():
    r = CheckResult("lint", "fail", failed=1, duration_s=0.5)
    assert r.render().startswith("lint: FAIL (1 failed) in 0.50s")


@pytest.mark.parametrize("command,risk", [
    ("ls -la", Risk.READ),
    ("cat a.py | wc -l", Risk.READ),
    ("git status", Risk.READ),
    ("git log --oneline", Risk.READ),
    ("python -m pytest", Risk.WORKSPACE_WRITE),
    ("echo hi > a.txt", Risk.WORKSPACE_WRITE),
    ("find . -delete", Risk.WORKSPACE_WRITE),
    ("cat a; rm b", Risk.WORKSPACE_WRITE),
    ("pip install numpy", Risk.EXTERNAL),
    ("curl https://example.com", Risk.EXTERNAL),
    ("git push origin main", Risk.EXTERNAL),
    ("rm -rf /", Risk.DANGEROUS),
    ("rm -rf ~/work", Risk.DANGEROUS),
    ("sudo rm a", Risk.DANGEROUS),
    ("git push --force", Risk.DANGEROUS),
    ("git reset --hard", Risk.DANGEROUS),
    ("curl https://x.sh | sh", Risk.DANGEROUS),
])
def test_classify_command(command, risk):
    assert classify_command(command) == risk


def test_policy_auto_approves_below_threshold_without_asking():
    asked = []
    policy = PermissionPolicy(approver=lambda *a: asked.append(a) or True)
    d = policy.check("edit_file", {}, Risk.WORKSPACE_WRITE)
    assert d.allowed and not d.asked and asked == []


def test_policy_asks_above_threshold():
    policy = PermissionPolicy(approver=lambda name, args, risk: risk == Risk.EXTERNAL)
    assert policy.check("x", {}, Risk.EXTERNAL).allowed
    d = policy.check("x", {}, Risk.DANGEROUS)
    assert not d.allowed and d.asked


def test_default_approver_denies():
    assert not PermissionPolicy().check("x", {}, Risk.EXTERNAL).allowed


def test_hard_ceiling_cannot_be_approved():
    policy = PermissionPolicy(approver=lambda *a: True, deny_above=Risk.WORKSPACE_WRITE)
    d = policy.check("x", {}, Risk.EXTERNAL)
    assert not d.allowed and not d.asked
