"""Repository search: text search (ripgrep, git grep, or pure Python), files, and symbols.

Text search answers "where does this string appear". The symbol index answers
"where is this function defined, and who uses it", which is what a model usually
means. Both are cheap enough to run on every turn.
"""

from __future__ import annotations

import ast
import fnmatch
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .fs_tools import IGNORED_DIRS, resolve
from .permissions import Risk
from .tools import ToolContext, ToolError, tool


@dataclass
class Match:
    path: str
    line: int
    text: str


def iter_files(root: Path, glob: str | None = None) -> list[Path]:
    """The file index: every non-ignored file under root, sorted for stable output."""
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or IGNORED_DIRS & set(rel.parts) or p.suffix in (".pyc", ".so"):
            continue
        if glob and not (fnmatch.fnmatch(str(rel), glob) or fnmatch.fnmatch(p.name, glob)):
            continue
        out.append(p)
    return out


def _parse_lines(stdout: str) -> list[Match]:
    out = []
    for line in stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[1].isdigit():
            out.append(Match(parts[0].removeprefix("./"), int(parts[1]), parts[2]))
    return out


def search_ripgrep(root: Path, pattern: str, glob: str | None = None) -> list[Match]:
    cmd = ["rg", "--line-number", "--no-heading", "--color", "never", "-e", pattern]
    if glob:
        cmd += ["-g", glob]
    res = subprocess.run(cmd + ["."], cwd=root, capture_output=True, text=True, timeout=30)
    if res.returncode > 1:
        raise ToolError(res.stderr.strip() or "ripgrep failed")
    return _parse_lines(res.stdout)


def search_gitgrep(root: Path, pattern: str, glob: str | None = None) -> list[Match]:
    cmd = ["git", "grep", "-n", "-I", "-E", "--untracked", "-e", pattern, "--"]
    cmd += [glob] if glob else ["."]
    res = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=30)
    if res.returncode > 1:
        raise ToolError(res.stderr.strip() or "git grep failed")
    return _parse_lines(res.stdout)


def search_python(root: Path, pattern: str, glob: str | None = None) -> list[Match]:
    """The reference implementation: slow, dependency-free, obviously correct."""
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"invalid regex {pattern!r}: {e}") from None
    out = []
    for p in iter_files(root, glob):
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue  # binary file
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                out.append(Match(str(p.relative_to(root)), i, line))
    return out


def pick_backend(root: Path) -> str:
    if shutil.which("rg"):
        return "ripgrep"
    if shutil.which("git") and (root / ".git").exists():
        return "git-grep"
    return "python"


BACKENDS = {"ripgrep": search_ripgrep, "git-grep": search_gitgrep, "python": search_python}


@tool(risk=Risk.READ)
def search(ctx: ToolContext, pattern: str, glob: str | None = None, max_results: int = 50) -> str:
    """Search file contents with a regular expression; returns path:line: text matches.

    Args:
        pattern: Regular expression, for example "def mean" or "TODO|FIXME".
        glob: Optional file filter, for example "*.py" or "tests/*".
        max_results: Maximum number of matching lines to return.
    """
    root = Path(ctx.workspace).resolve()
    matches = BACKENDS[pick_backend(root)](root, pattern, glob)
    if not matches:
        return f"no matches for {pattern!r}"
    shown = [f"{m.path}:{m.line}: {m.text.strip()[:200]}" for m in matches[:max_results]]
    if len(matches) > max_results:
        shown.append(f"[{len(matches) - max_results} more matches; narrow the pattern or glob]")
    return "\n".join(shown)


@tool(risk=Risk.READ)
def find_files(ctx: ToolContext, glob: str) -> str:
    """List files whose path or name matches a glob, for example "*.py" or "tests/test_*".

    Args:
        glob: Shell-style pattern matched against the relative path and the file name.
    """
    root = Path(ctx.workspace).resolve()
    files = [str(p.relative_to(root)) for p in iter_files(root, glob)]
    return "\n".join(files[:200]) or f"no files match {glob!r}"


# ---------------------------------------------------------------- symbol index

@dataclass
class Symbol:
    name: str          # qualified, e.g. "Stats.mean"
    kind: str          # "class" | "function" | "method"
    path: str
    line: int
    signature: str


@dataclass
class FileSymbols:
    path: str
    imports: list[str] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    names_used: set[str] = field(default_factory=set)


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """`def name(args) -> ret` without the body, as the model would want to read it."""
    args = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"def {node.name}({args}){ret}"


def index_file(path: Path, root: Path) -> FileSymbols:
    rel = str(path.relative_to(root))
    fs = FileSymbols(rel)
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return fs
    for node in tree.body:
        if isinstance(node, ast.Import):
            fs.imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            fs.imports.append("." * node.level + (node.module or ""))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fs.symbols.append(Symbol(node.name, "function", rel, node.lineno, signature(node)))
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            fs.symbols.append(Symbol(node.name, "class", rel, node.lineno,
                                     f"class {node.name}({bases})" if bases else f"class {node.name}"))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fs.symbols.append(Symbol(f"{node.name}.{item.name}", "method", rel,
                                             item.lineno, signature(item)))
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            fs.names_used.add(n.id)
        elif isinstance(n, ast.Attribute):
            fs.names_used.add(n.attr)
    return fs


class SymbolIndex:
    """Definitions and cross-file references for every Python file in the workspace."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.files = {fs.path: fs for fs in
                      (index_file(p, self.root) for p in iter_files(self.root, "*.py"))}

    def definitions(self, name: str) -> list[Symbol]:
        return [s for fs in self.files.values() for s in fs.symbols
                if s.name == name or s.name.split(".")[-1] == name]

    def referencing_files(self, name: str) -> list[str]:
        short = name.split(".")[-1]
        return [p for p, fs in self.files.items() if short in fs.names_used]


@tool(risk=Risk.READ)
def find_symbol(ctx: ToolContext, name: str) -> str:
    """Find where a class, function or method is defined and which files use it.

    Args:
        name: Symbol name, for example "mean" or "Stats.mean".
    """
    idx = SymbolIndex(resolve(ctx, "."))
    defs = idx.definitions(name)
    if not defs:
        return f"no definition of {name!r} found"
    lines = [f"{s.path}:{s.line}: {s.signature}" for s in defs]
    users = idx.referencing_files(name)
    lines.append("used in: " + (", ".join(users) if users else "(no other references)"))
    return "\n".join(lines)
