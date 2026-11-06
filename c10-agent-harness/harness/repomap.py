"""The repository map: the whole codebase as a few hundred tokens of signatures.

    stats.py
     ├ import math
     ├ def mean(xs: list[float]) -> float
     └ class Summary
        └ def describe(self) -> str

Files are ranked by how often other files use their symbols (a one-step version of
the PageRank ranking Aider uses), plus a boost for names the task mentions, and the
map is cut to a token budget. The model gets an outline first and reads files on demand.
"""

from __future__ import annotations

import re
from pathlib import Path

from .search import FileSymbols, SymbolIndex
from .tokens import count_tokens
from .tools import ToolContext, tool


def rank_files(idx: SymbolIndex, focus: str = "") -> list[tuple[str, float]]:
    """Score = cross-file references to this file's symbols + 5 per symbol named in `focus`."""
    words = set(re.findall(r"\w+", focus.lower()))
    scores: dict[str, float] = {}
    for path, fs in idx.files.items():
        defined = {s.name.split(".")[-1] for s in fs.symbols}
        refs = sum(len(defined & other.names_used) for p, other in idx.files.items() if p != path)
        stem_hits = sum(w in words for w in re.findall(r"\w+", Path(path).stem.lower()))
        named = sum(n.lower() in words for n in defined)
        scores[path] = refs + 5 * (named + stem_hits)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def render_file(fs: FileSymbols) -> str:
    items = [("import " + ", ".join(fs.imports), 0)] if fs.imports else []
    items += [(s.signature, s.name.count(".")) for s in fs.symbols]
    def is_last(i: int) -> bool:
        for _, d in items[i + 1:]:
            if d <= items[i][1]:
                return d < items[i][1]
        return True

    lines = [fs.path]
    parent_last = True
    for i, (text, depth) in enumerate(items):
        if depth == 0:
            parent_last = is_last(i)
            lines.append(" " + ("└ " if parent_last else "├ ") + text)
        else:  # a method: draw the parent's vertical rule unless the class was the last item
            rule = "   " if parent_last else "│  "
            lines.append(" " + rule + ("└ " if is_last(i) else "├ ") + text)
    return "\n".join(lines)


def build_repo_map(root: Path, max_tokens: int = 1024, focus: str = "") -> str:
    """Render ranked file outlines until the token budget is spent."""
    idx = SymbolIndex(root)
    blocks, used, skipped = [], 0, 0
    for path, _ in rank_files(idx, focus):
        block = render_file(idx.files[path])
        cost = count_tokens(block)
        if used + cost > max_tokens:
            skipped += 1
            continue
        blocks.append(block)
        used += cost
    if skipped:
        blocks.append(f"[{skipped} more files not shown; use find_files or search]")
    return "\n".join(blocks)


@tool
def repo_map(ctx: ToolContext, focus: str = "", max_tokens: int = 1024) -> str:
    """Show a compact outline of the repository: files, imports, classes and function signatures.

    Args:
        focus: Words from the task; files defining matching names are shown first.
        max_tokens: Size budget for the map.
    """
    return build_repo_map(Path(ctx.workspace), max_tokens, focus)
