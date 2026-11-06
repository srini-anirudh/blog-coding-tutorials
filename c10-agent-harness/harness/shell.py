"""The shell tool: run a command in the sandbox and report what happened, compactly.

The model needs four facts from a command: did it finish, the exit code, what went
to stdout, what went to stderr. Everything else (timing, byte counts, truncation)
is reported in one footer line so it never crowds the actual output.
"""

from __future__ import annotations

from pathlib import Path

from .permissions import classify_command
from .sandbox import ExecResult
from .tools import ToolContext, ToolError, ToolOutput, tool
from .truncation import truncate_output

_spill_counter = 0


def format_exec(res: ExecResult, workspace: Path | None = None, max_lines: int = 120) -> str:
    """Render an ExecResult for the model, truncating each stream separately."""
    global _spill_counter
    parts = []
    if res.timed_out:
        parts.append(f"TIMED OUT after {res.duration_s:.1f}s; the process group was killed.")
    elif res.killed_reason == "memory":
        parts.append("KILLED: memory limit exceeded; the process group was killed.")
    parts.append(f"exit code: {res.exit_code}")
    for name, text in (("stdout", res.stdout), ("stderr", res.stderr)):
        if not text.strip():
            continue
        spill = name_rel = None
        if workspace is not None and len(text.splitlines()) > max_lines:
            _spill_counter += 1
            name_rel = f".harness/outputs/{_spill_counter:03d}-{name}.txt"
            spill = Path(workspace) / name_rel
        t = truncate_output(text, max_lines=max_lines, spill_path=spill, spill_name=name_rel)
        parts.append(f"--- {name} ---\n{t.text}")
    parts.append(f"[{res.duration_s:.2f}s, {res.stdout_bytes} B stdout, {res.stderr_bytes} B stderr]")
    return "\n".join(parts)


@tool(risk=lambda args: classify_command(args.get("command", "")))
def run_command(ctx: ToolContext, command: str, timeout: int = 60, cwd: str = ".") -> ToolOutput:
    """Run a shell command in the workspace sandbox and return exit code, stdout and stderr.

    Args:
        command: The shell command line, for example "python -m pytest -x -q".
        timeout: Seconds before the whole process group is killed.
        cwd: Working directory relative to the workspace root.
    """
    if ctx.sandbox is None:
        raise ToolError("no sandbox configured for shell commands")
    try:
        res = ctx.sandbox.run(command, timeout=timeout, cwd=cwd)
    except ValueError as e:  # cwd escaping the workspace
        raise ToolError(str(e)) from None
    return ToolOutput(format_exec(res, ctx.workspace), is_error=not res.ok,
                      data={"exit_code": res.exit_code, "timed_out": res.timed_out,
                            "duration_s": round(res.duration_s, 3)})
