"""Verification tools: tests, lint, type check, build. One normalised result shape for all.

Different tools print failures in different dialects. The model should not have to
learn four dialects to answer one question: what is broken, and where. So every
check returns a CheckResult with the same fields, and renders it the same way.
"""

from __future__ import annotations

import ast
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .fs_tools import IGNORED_DIRS, resolve
from .permissions import Risk
from .sandbox import ExecResult
from .tools import ToolContext, ToolError, ToolOutput, tool
from .truncation import truncate_output


@dataclass
class Diagnostic:
    file: str
    line: int | None
    code: str            # test id for tests, rule code for linters
    message: str


@dataclass
class CheckResult:
    check: str           # "tests" | "lint" | "typecheck" | "build"
    status: str          # "pass" | "fail" | "error" | "timeout" | "unavailable"
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    diagnostics: list[Diagnostic] = field(default_factory=list)
    details: str = ""    # trimmed raw output (tracebacks) for context
    duration_s: float = 0.0
    command: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def render(self, max_diagnostics: int = 20) -> str:
        counts = ", ".join(f"{v} {k}" for k, v in
                           (("passed", self.passed), ("failed", self.failed),
                            ("errors", self.errors), ("skipped", self.skipped)) if v)
        lines = [f"{self.check}: {self.status.upper()}" + (f" ({counts})" if counts else "")
                 + f" in {self.duration_s:.2f}s"]
        for d in self.diagnostics[:max_diagnostics]:
            loc = f"{d.file}:{d.line}" if d.line else d.file
            lines.append(f"  {loc} [{d.code}] {d.message}")
        if len(self.diagnostics) > max_diagnostics:
            lines.append(f"  ... {len(self.diagnostics) - max_diagnostics} more")
        if self.details:
            lines.append(self.details)
        return "\n".join(lines)

    def to_output(self) -> ToolOutput:
        return ToolOutput(self.render(), is_error=not self.ok, kind="test", data=asdict(self))


# ---------------------------------------------------------------- pytest

_SUMMARY_RE = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")
_SHORT_RE = re.compile(r"^(FAILED|ERROR) (\S+?)(?:::(\S+))?(?: - (.*))?$")
_LOCATION_RE = re.compile(r"^(\S+\.py):(\d+): in (\w+)", re.M)


def parse_pytest(res: ExecResult) -> CheckResult:
    out = res.stdout + "\n" + res.stderr
    r = CheckResult("tests", "pass", duration_s=res.duration_s, command=res.command)
    tail = "\n".join(out.strip().splitlines()[-3:])
    for n, kind in _SUMMARY_RE.findall(tail):
        n = int(n)
        if kind == "passed":
            r.passed = n
        elif kind == "failed":
            r.failed = n
        elif kind.startswith("error"):
            r.errors = n
        elif kind == "skipped":
            r.skipped = n
    # "tests/test_x.py:15: in test_name" lines inside each failure section
    locations = {name: (f, int(ln)) for f, ln, name in _LOCATION_RE.findall(out)}
    for line in out.splitlines():
        m = _SHORT_RE.match(line.strip())
        if m:
            kind, file, test, msg = m.groups()
            lineno = locations.get((test or "").split("[")[0], (file, None))[1]
            r.diagnostics.append(Diagnostic(file, lineno, test or kind.lower(), msg or kind))
    if res.timed_out:
        r.status = "timeout"
    elif res.exit_code == 0:
        r.status = "pass"
    elif res.exit_code == 1:
        r.status = "fail"
    elif res.exit_code == 5:
        r.status = "error"
        r.details = "no tests were collected"
    else:
        r.status = "error"
    if r.status != "pass":
        # Keep the "E   ..." assertion lines and locations; drop the passing noise.
        r.details = r.details or truncate_output(_failure_section(out), max_lines=60).text
    return r


def _failure_section(out: str) -> str:
    """The part of pytest output between the FAILURES banner and the short summary."""
    m = re.search(r"^=+ (FAILURES|ERRORS) =+$", out, re.M)
    end = re.search(r"^=+ short test summary info =+$", out, re.M)
    if not m:
        return out.strip()
    return out[m.start(): end.start() if end else len(out)].strip()


@tool(risk=Risk.WORKSPACE_WRITE)
def run_tests(ctx: ToolContext, target: str = "", timeout: int = 120) -> ToolOutput:
    """Run the test suite (pytest) and return a normalised pass/fail summary.

    Args:
        target: Optional test path or node id, for example "tests/test_stats.py::test_mean".
        timeout: Seconds before the test run is killed.
    """
    if ctx.sandbox is None:
        raise ToolError("no sandbox configured")
    cmd = f"{shlex.quote(ctx.sandbox.python)} -m pytest -q -rfE --tb=short -p no:cacheprovider"
    if target:
        cmd += " " + shlex.quote(target)
    res = ctx.sandbox.run(cmd, timeout=timeout)
    if "No module named pytest" in res.stderr:  # not a test failure: nothing ran
        return CheckResult("tests", "unavailable", details="pytest is not installed in this sandbox",
                           duration_s=res.duration_s, command=res.command).to_output()
    return parse_pytest(res).to_output()


# ---------------------------------------------------------------- lint

_CONCISE_RE = re.compile(r"^(.+?):(\d+):(?:\d+:)? ([A-Z]+\d+) (.*)$")


def builtin_lint(root: Path) -> list[Diagnostic]:
    """A tiny pyflakes: syntax errors (E999) and unused imports (F401). No dependencies."""
    diags: list[Diagnostic] = []
    for path in sorted(root.rglob("*.py")):
        if IGNORED_DIRS & set(path.relative_to(root).parts):
            continue
        rel = str(path.relative_to(root))
        src = path.read_text(errors="replace")
        try:
            tree = ast.parse(src, rel)
        except SyntaxError as e:
            diags.append(Diagnostic(rel, e.lineno, "E999", f"SyntaxError: {e.msg}"))
            continue
        imported: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    name = (a.asname or a.name).split(".")[0]
                    if name != "*":
                        imported[name] = node.lineno
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.value.id for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
        exported = set()
        for node in ast.walk(tree):  # names listed in __all__ count as used
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
                exported |= {e.value for e in getattr(node.value, "elts", []) if isinstance(e, ast.Constant)}
        if path.name == "__init__.py":
            continue  # re-exports are the point of __init__
        for name, line in imported.items():
            if name not in used and name not in exported and name != "annotations":
                diags.append(Diagnostic(rel, line, "F401", f"'{name}' imported but unused"))
    return diags


@tool(risk=Risk.READ)
def run_linter(ctx: ToolContext, path: str = ".") -> ToolOutput:
    """Lint Python files (ruff if installed, otherwise a built-in checker) and list problems.

    Args:
        path: File or directory relative to the workspace root.
    """
    target = resolve(ctx, path)
    probe = ctx.sandbox.run(f"{shlex.quote(ctx.sandbox.python)} -m ruff --version", timeout=20)
    if probe.ok:
        res = ctx.sandbox.run(f"{shlex.quote(ctx.sandbox.python)} -m ruff check --output-format=concise "
                              f"{shlex.quote(path)}", timeout=60)
        diags = [Diagnostic(m[1], int(m[2]), m[3], m[4]) for m in
                 (_CONCISE_RE.match(x) for x in res.stdout.splitlines()) if m]
        r = CheckResult("lint", "pass" if res.exit_code == 0 else "fail", diagnostics=diags,
                        duration_s=res.duration_s, command=res.command)
    else:
        import time
        t0 = time.monotonic()
        root = Path(ctx.workspace).resolve()
        diags = builtin_lint(target if target.is_dir() else target.parent)
        if target.is_file():
            diags = [d for d in diags if Path(d.file).name == target.name]
        prefix = str(target.relative_to(root)) if target.is_dir() and target != root else ""
        for d in diags:
            d.file = f"{prefix}/{d.file}" if prefix else d.file
        r = CheckResult("lint", "fail" if diags else "pass", diagnostics=diags,
                        duration_s=time.monotonic() - t0, command="builtin_lint")
    r.failed = len(r.diagnostics)
    return r.to_output()


# ---------------------------------------------------------------- type check and build

_MYPY_RE = re.compile(r"^(.+?):(\d+): (error|note): (.*?)(?:\s+\[([\w-]+)\])?$")


@tool(risk=Risk.READ)
def run_typecheck(ctx: ToolContext, path: str = ".") -> ToolOutput:
    """Type-check with mypy if it is installed; reports 'unavailable' otherwise.

    Args:
        path: File or directory relative to the workspace root.
    """
    py = shlex.quote(ctx.sandbox.python)
    if not ctx.sandbox.run(f"{py} -m mypy --version", timeout=20).ok:
        return CheckResult("typecheck", "unavailable",
                           details="mypy is not installed in this sandbox").to_output()
    res = ctx.sandbox.run(f"{py} -m mypy --no-error-summary --hide-error-context "
                          f"{shlex.quote(path)}", timeout=180)
    diags = [Diagnostic(m[1], int(m[2]), m[5] or m[3], m[4]) for m in
             (_MYPY_RE.match(x) for x in res.stdout.splitlines()) if m and m[3] == "error"]
    return CheckResult("typecheck", "pass" if res.exit_code == 0 else "fail", failed=len(diags),
                       diagnostics=diags, duration_s=res.duration_s, command=res.command).to_output()


_COMPILE_RE = re.compile(r'File "(.+?)", line (\d+)')


@tool(risk=Risk.WORKSPACE_WRITE)
def build(ctx: ToolContext, path: str = ".") -> ToolOutput:
    """Byte-compile every Python file (the 'build' step for a Python project).

    Args:
        path: Directory relative to the workspace root.
    """
    resolve(ctx, path)
    res = ctx.sandbox.run(f"{shlex.quote(ctx.sandbox.python)} -m compileall -q {shlex.quote(path)}",
                          timeout=120)
    out = res.stdout + res.stderr
    diags = [Diagnostic(f, int(ln), "compile", "compile error") for f, ln in _COMPILE_RE.findall(out)]
    return CheckResult("build", "pass" if res.ok else "fail", failed=len(diags), diagnostics=diags,
                       details="" if res.ok else truncate_output(out, max_lines=40).text,
                       duration_s=res.duration_s, command=res.command).to_output()
