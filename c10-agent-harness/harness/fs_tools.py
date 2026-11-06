"""Filesystem tools: read, list, write. Every path is checked against the workspace root.

The FileTracker enforces read-before-write. An edit based on a stale view of a
file (the model read it, something else changed it, the model edits from memory)
is the most common silent corruption in coding agents. We refuse such edits and
say why, which costs one extra read instead of a broken file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .permissions import Risk
from .tools import ToolContext, ToolError, ToolOutput, tool

IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache",
                ".pytest_cache", ".harness", ".ruff_cache", "dist", "build"}


def resolve(ctx: ToolContext, path: str) -> Path:
    """Workspace-relative path to an absolute one; refuse anything that escapes the root."""
    root = Path(ctx.workspace).resolve()
    full = (root / path).resolve()
    if not full.is_relative_to(root):
        raise ToolError(f"path {path!r} is outside the workspace")
    return full


def digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


@dataclass
class FileTracker:
    """Remembers the content hash of every file the model has seen."""

    seen: dict[str, str] = field(default_factory=dict)

    def mark_read(self, path: Path, text: str) -> None:
        self.seen[str(path)] = digest(text)

    def check_fresh(self, path: Path) -> None:
        """Raise unless the model's last view of `path` matches what is on disk."""
        if not path.exists():
            return  # creating a new file needs no prior read
        key = str(path)
        if key not in self.seen:
            raise ToolError(f"read {path.name} before editing it")
        if self.seen[key] != digest(path.read_text()):
            raise ToolError(f"{path.name} changed since you last read it; read it again before editing")


def number_lines(text: str, start: int = 1) -> str:
    return "\n".join(f"{i:>5} | {line}" for i, line in enumerate(text.splitlines(), start))


@tool(risk=Risk.READ)
def read_file(ctx: ToolContext, path: str, start: int = 1, end: int | None = None) -> str:
    """Read a text file and return it with line numbers.

    Args:
        path: File path relative to the workspace root.
        start: First line to show (1-based).
        end: Last line to show (inclusive). Omit to read up to 400 lines.
    """
    full = resolve(ctx, path)
    if not full.is_file():
        raise ToolError(f"no such file: {path}")
    text = full.read_text(errors="replace")
    if ctx.files is not None:
        ctx.files.mark_read(full, text)
    lines = text.splitlines()
    end = min(end or start + 399, len(lines))
    body = number_lines("\n".join(lines[start - 1:end]), start)
    more = f"\n[lines {start}-{end} of {len(lines)}]" if end < len(lines) or start > 1 else ""
    return body + more


@tool(risk=Risk.READ)
def list_dir(ctx: ToolContext, path: str = ".", depth: int = 2) -> str:
    """List files and folders as an indented tree, skipping caches and virtualenvs.

    Args:
        path: Directory relative to the workspace root.
        depth: How many levels to descend.
    """
    root = resolve(ctx, path)
    if not root.is_dir():
        raise ToolError(f"not a directory: {path}")
    out: list[str] = []

    def walk(d: Path, level: int) -> None:
        for child in sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if child.name in IGNORED_DIRS or child.name.endswith(".pyc"):
                continue
            out.append("  " * level + child.name + ("/" if child.is_dir() else ""))
            if child.is_dir() and level + 1 < depth:
                walk(child, level + 1)

    walk(root, 0)
    return "\n".join(out) or "(empty)"


@tool(risk=Risk.WORKSPACE_WRITE)
def write_file(ctx: ToolContext, path: str, content: str) -> ToolOutput:
    """Create a file or replace its whole content (whole-file rewrite).

    Args:
        path: File path relative to the workspace root.
        content: The complete new file content.
    """
    from .editing import make_diff

    full = resolve(ctx, path)
    if ctx.files is not None:
        ctx.files.check_fresh(full)
    old = full.read_text() if full.exists() else ""
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    if ctx.files is not None:
        ctx.files.mark_read(full, content)
    diff = make_diff(old, content, path)
    return ToolOutput(f"wrote {path} ({len(content.splitlines())} lines)\n{diff}",
                      kind="patch", data={"path": path, "diff": diff, "format": "whole_file"})
