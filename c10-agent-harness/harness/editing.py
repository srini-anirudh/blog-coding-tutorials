"""Edit formats and a patch applier that fails loudly instead of corrupting files.

Four ways a model can express an edit:

  whole file      write the full new content          simple, expensive, silently drops code
  search/replace  quote the old block, give the new   cheap, needs a unique quote
  strict diff     unified diff, line numbers exact    what `git apply` does with no fuzz
  anchored diff   unified diff, numbers are hints     context lines locate the hunk

The applier tries an exact match first, then a whitespace-tolerant one, and when it
still fails it shows the model the closest region of the file so the retry is informed.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .fs_tools import resolve
from .permissions import Risk
from .tools import ToolContext, ToolError, ToolOutput, tool


class EditError(ToolError):
    """An edit that could not be applied. The message is written for the model."""


def make_diff(old: str, new: str, path: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        f"a/{path}", f"b/{path}", n=3))


# ---------------------------------------------------------------- block matching

def _norm(line: str, level: int) -> str:
    """level 0: exact. level 1: ignore trailing spaces. level 2: ignore indentation too."""
    return line if level == 0 else line.rstrip() if level == 1 else line.strip()


def find_block(lines: list[str], block: list[str], level: int) -> list[int]:
    """All start indices where `block` occurs in `lines` under normalisation `level`."""
    if not block:
        return []
    target = [_norm(b, level) for b in block]
    first = target[0]
    hits = [i for i in range(len(lines) - len(block) + 1)
            if _norm(lines[i], level) == first
            and all(_norm(lines[i + k], level) == target[k] for k in range(1, len(block)))]
    if level == 2:  # only a uniform shift is safe to repair; flattened structure is refused
        hits = [i for i in hits if _uniform_shift(block, lines[i:i + len(block)])]
    return hits


def _uniform_shift(block: list[str], matched: list[str]) -> bool:
    """True if every line of `block` is indented differently from the file by the same amount."""
    return len({len(_indent(m)) - len(_indent(b)) for b, m in zip(block, matched) if b.strip()}) <= 1


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def reindent(new: list[str], old_block: list[str], matched: list[str]) -> list[str]:
    """Shift `new` by the indentation difference between what the model quoted and the file."""
    src = next((_indent(x) for x in old_block if x.strip()), "")
    dst = next((_indent(x) for x in matched if x.strip()), "")
    if src == dst:
        return new
    out = []
    for line in new:
        if line.strip() and line.startswith(src):
            out.append(dst + line[len(src):])
        else:
            out.append(line)
    return out


def closest_region(lines: list[str], block: list[str]) -> str:
    """The window of the file most similar to `block`, with line numbers, for the error message."""
    if not lines or not block:
        return ""
    n = len(block)
    want = "\n".join(b.strip() for b in block)
    best, best_i = 0.0, 0
    for i in range(max(1, len(lines) - n + 1)):
        got = "\n".join(x.strip() for x in lines[i:i + n])
        r = difflib.SequenceMatcher(None, want, got).ratio()
        if r > best:
            best, best_i = r, i
    region = lines[best_i:best_i + n]
    numbered = "\n".join(f"{best_i + 1 + k:>5} | {x}" for k, x in enumerate(region))
    return f"closest match ({best:.0%} similar) at lines {best_i + 1}-{best_i + n}:\n{numbered}"


@dataclass
class Applied:
    text: str
    strategy: str          # "exact" | "trailing-whitespace" | "indentation" | "line-number"
    hunks: int = 1


def _splice(lines: list[str], i: int, n: int, new: list[str]) -> list[str]:
    return lines[:i] + new + lines[i + n:]


def _join(lines: list[str], had_newline: bool) -> str:
    return "\n".join(lines) + ("\n" if had_newline and lines else "")


# ---------------------------------------------------------------- search / replace

LEVEL_NAMES = ["exact", "trailing-whitespace", "indentation"]


def apply_search_replace(text: str, search: str, replace: str, replace_all: bool = False) -> Applied:
    """Replace the unique occurrence of `search` with `replace`."""
    lines = text.splitlines()
    old, new = search.splitlines(), replace.splitlines()
    if not old:
        raise EditError("search text is empty; quote the lines you want to replace")
    for level in range(3):
        hits = find_block(lines, old, level)
        if not hits:
            continue
        if len(hits) > 1 and not replace_all:
            where = ", ".join(str(h + 1) for h in hits)
            raise EditError(f"search text matches {len(hits)} places (lines {where}); "
                            "add surrounding lines to make it unique, or set replace_all")
        for i in reversed(hits):  # back to front so earlier indices stay valid
            block = new if level < 2 else reindent(new, old, lines[i:i + len(old)])
            lines = _splice(lines, i, len(old), block)
        return Applied(_join(lines, text.endswith("\n")), LEVEL_NAMES[level], len(hits))
    raise EditError("search text not found in file.\n" + closest_region(lines, old))


# ---------------------------------------------------------------- unified diff

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    old_start: int                      # 1-based line number the diff claims
    old: list[str] = field(default_factory=list)   # context + removed lines
    new: list[str] = field(default_factory=list)   # context + added lines


@dataclass
class FilePatch:
    path: str
    hunks: list[Hunk]
    is_new: bool = False


def _strip_prefix(path: str) -> str:
    path = path.split("\t")[0].strip()
    return path[2:] if path[:2] in ("a/", "b/") else path


def parse_unified_diff(diff: str) -> list[FilePatch]:
    """Parse one or more file sections of a unified diff."""
    patches: list[FilePatch] = []
    cur: FilePatch | None = None
    hunk: Hunk | None = None
    lines = diff.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            old_path, new_path = line[4:], lines[i + 1][4:]
            is_new = old_path.strip() == "/dev/null"
            cur = FilePatch(_strip_prefix(new_path), [], is_new)
            patches.append(cur)
            hunk = None
            i += 2
            continue
        m = HUNK_RE.match(line)
        if m:
            if cur is None:
                raise EditError("hunk header before any ---/+++ file header")
            hunk = Hunk(int(m.group(1)))
            cur.hunks.append(hunk)
        elif hunk is not None and line[:1] in (" ", "-", "+"):
            body = line[1:]
            if line[0] != "+":
                hunk.old.append(body)
            if line[0] != "-":
                hunk.new.append(body)
        elif hunk is not None and line == "":
            hunk.old.append("")   # editors strip the single space of blank context lines
            hunk.new.append("")
        elif line.startswith("\\ No newline"):
            pass
        i += 1
    if not patches:
        raise EditError("no file headers found; a unified diff starts with '--- a/path' and '+++ b/path'")
    for p in patches:
        if not p.hunks:
            raise EditError(f"no @@ hunks for {p.path}")
    return patches


def apply_hunks(text: str, hunks: list[Hunk], strict: bool = False) -> Applied:
    """Apply hunks in order. strict=True: old lines must sit exactly at old_start (no fuzz).

    Anchored mode treats old_start as a hint: it searches for the hunk's old lines
    (exact, then whitespace-tolerant) and takes the occurrence nearest the hint.
    """
    lines = text.splitlines()
    offset = 0          # how far earlier hunks moved later line numbers
    used: list[str] = []
    for k, h in enumerate(hunks, 1):
        hint = h.old_start - 1 + offset
        if not h.old:   # pure insertion: only the line number can place it
            at = max(0, min(h.old_start + offset, len(lines)))
            lines = _splice(lines, at, 0, h.new)
            offset += len(h.new)
            continue
        if strict:
            if lines[hint:hint + len(h.old)] != h.old:
                raise EditError(f"hunk {k} does not match at line {h.old_start} (strict mode).\n"
                                + closest_region(lines, h.old))
            at, level = hint, 0
        else:
            at = level = None
            for lvl in range(3):
                hits = find_block(lines, h.old, lvl)
                if hits:
                    at, level = min(hits, key=lambda s: abs(s - hint)), lvl
                    break
            if at is None:
                raise EditError(f"hunk {k} context not found in file.\n" + closest_region(lines, h.old))
        block = h.new if level < 2 else reindent(h.new, h.old, lines[at:at + len(h.old)])
        lines = _splice(lines, at, len(h.old), block)
        offset += len(h.new) - len(h.old) + (at - hint)
        used.append(LEVEL_NAMES[level] if at == hint else f"{LEVEL_NAMES[level]}, moved {at - hint:+d} lines")
    strategy = "; ".join(dict.fromkeys(used)) or "insert"
    return Applied(_join(lines, text.endswith("\n") or not text), strategy, len(hunks))


# ---------------------------------------------------------------- the tools

@tool(risk=Risk.WORKSPACE_WRITE)
def edit_file(ctx: ToolContext, path: str, search: str, replace: str, replace_all: bool = False) -> ToolOutput:
    """Replace an exact block of lines in a file (search/replace edit).

    Args:
        path: File path relative to the workspace root.
        search: The existing lines to replace, copied exactly, with enough context to be unique.
        replace: The new lines.
        replace_all: Replace every occurrence instead of requiring a unique match.
    """
    full = resolve(ctx, path)
    if not full.is_file():
        raise EditError(f"no such file: {path}")
    if ctx.files is not None:
        ctx.files.check_fresh(full)
    old = full.read_text()
    applied = apply_search_replace(old, search, replace, replace_all)
    full.write_text(applied.text)
    if ctx.files is not None:
        ctx.files.mark_read(full, applied.text)
    diff = make_diff(old, applied.text, path)
    return ToolOutput(f"edited {path} (match: {applied.strategy})\n{diff}", kind="patch",
                      data={"path": path, "diff": diff, "format": "search_replace",
                            "strategy": applied.strategy})


@tool(risk=Risk.WORKSPACE_WRITE)
def apply_patch(ctx: ToolContext, patch: str) -> ToolOutput:
    """Apply a unified diff to one or more files. Line numbers are hints; context lines must match.

    Args:
        patch: A unified diff with '--- a/path' / '+++ b/path' headers and @@ hunks.
    """
    results, diffs = [], []
    staged: list[tuple] = []
    for fp in parse_unified_diff(patch):   # apply all files or none
        full = resolve(ctx, fp.path)
        if fp.is_new:
            old = ""
        else:
            if not full.is_file():
                raise EditError(f"no such file: {fp.path}")
            if ctx.files is not None:
                ctx.files.check_fresh(full)
            old = full.read_text()
        applied = apply_hunks(old, fp.hunks)
        staged.append((full, fp.path, old, applied))
    for full, path, old, applied in staged:
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(applied.text)
        if ctx.files is not None:
            ctx.files.mark_read(full, applied.text)
        diffs.append(make_diff(old, applied.text, path))
        results.append(f"patched {path}: {applied.hunks} hunk(s), match: {applied.strategy}")
    diff = "".join(diffs)
    return ToolOutput("\n".join(results) + "\n" + diff, kind="patch",
                      data={"path": ",".join(p for _, p, _, _ in staged), "diff": diff,
                            "format": "unified_diff"})
