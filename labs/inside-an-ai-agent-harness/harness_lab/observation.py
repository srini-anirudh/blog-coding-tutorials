"""Observation design (post, section 9): o_model = Phi(o_env).

A tool produced far more text than the context can take. Each policy below is one
choice of Phi with the same token budget. We ask one question of each: does the
line that explains the problem survive into the model's context?

    everything       the whole output
    first N          as many leading lines as fit
    head and tail    half the budget from each end
    summary          a per-tool parser: counts plus the lines it thinks matter
    spill to handle  the summary, plus one follow-up query against the stored output
    C10 rescue       C10's head + tail + rescued error lines (harness.truncation)
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from harness.tokens import count_tokens
from harness.truncation import truncate_output

POLICIES = ("everything", "first N", "head and tail", "summary", "spill to handle", "C10 rescue")


def _fit(lines: list[str], budget: int) -> list[str]:
    out, used = [], 0
    for ln in lines:
        c = count_tokens(ln) + 1
        if used + c > budget:
            break
        out.append(ln)
        used += c
    return out


def first_n(text: str, budget: int) -> str:
    lines = text.splitlines()
    kept = _fit(lines, budget - 12)
    return "\n".join(kept) + f"\n[{len(lines) - len(kept)} more lines]"


def head_and_tail(text: str, budget: int) -> str:
    lines = text.splitlines()
    head = _fit(lines, budget // 2 - 6)
    tail = _fit(lines[::-1], budget // 2 - 6)[::-1]
    if len(head) + len(tail) >= len(lines):
        return text
    return "\n".join(head + [f"... [{len(lines) - len(head) - len(tail)} lines omitted] ..."] + tail)


ERROR = re.compile(r"(^E\s|error[:\s]|Error\b|FAILED|Traceback|assert)", re.I)


def summary(text: str, kind: str, budget: int) -> str:
    """A parser per tool. grep: matches per file. Test and build logs: error-looking lines."""
    lines = text.splitlines()
    if kind == "grep":
        per_file = Counter(ln.split(":", 1)[0] for ln in lines if ":" in ln)
        head = [f"{len(lines)} matches in {len(per_file)} files"]
        return "\n".join(_fit(head + [f"{f}: {n}" for f, n in per_file.most_common()], budget))
    errs = [ln for ln in lines if ERROR.search(ln)]
    return "\n".join(_fit([f"{len(lines)} lines, {len(errs)} error-looking lines"] + errs + lines[-3:], budget))


def spill(text: str, kind: str, budget: int, query: str) -> str:
    """Summary plus a handle. The agent then asks once: lines matching `query`."""
    first = summary(text, kind, budget // 2) + "\n[full output saved as handle out-1; query it with grep]"
    hits = [ln for ln in text.splitlines() if re.search(query, ln)]
    second = f"grep {query!r} out-1: {len(hits)} lines\n" + "\n".join(_fit(hits, budget // 2))
    return first + "\n" + second


def apply_policy(policy: str, text: str, kind: str, budget: int, query: str) -> str:
    if policy == "everything":
        return text
    if policy == "first N":
        return first_n(text, budget)
    if policy == "head and tail":
        return head_and_tail(text, budget)
    if policy == "summary":
        return summary(text, kind, budget)
    if policy == "spill to handle":
        return spill(text, kind, budget, query)
    if policy == "C10 rescue":
        return truncate_output(text, max_lines=budget // 12).text
    raise ValueError(policy)


# ------------------------------------------------------------------ three real outputs

def grep_output(root: Path, n_files: int, key_at: float) -> tuple[str, str]:
    """Run real grep over n generated files full of TODOs. One TODO is the answer."""
    key = "TODO(security): session token is written to the debug log in plaintext"
    key_file = int(key_at * (n_files - 1))
    for i in range(n_files):
        body = [f"x_{j} = {j}  # TODO: tidy name {j}" if j % 3 == 0 else f"y_{j} = {j}" for j in range(60)]
        if i == key_file:
            body[31] = f"log.debug(token)  # {key}"
        (root / f"m{i:04d}.py").write_text("\n".join(body) + "\n")
    out = subprocess.run(["grep", "-rn", "TODO", "."], cwd=root, capture_output=True, text=True).stdout
    return "\n".join(sorted(out.splitlines())), key


def pytest_output(root: Path, n_tests: int, fail_at: int) -> tuple[str, str]:
    """Run real pytest, with -s so every test's prints go to the output."""
    src = ["def test_{i}():\n    for k in range(12):\n        print('step', k, 'of test {i}: ok')\n"
           .format(i=i) for i in range(n_tests)]
    src[fail_at] = (f"def test_{fail_at}():\n    for k in range(12):\n        print('step', k)\n"
                    f"    expected, got = 42, 41\n    assert got == expected\n")
    (root / "test_gen.py").write_text("\n".join(src))
    res = subprocess.run([sys.executable, "-m", "pytest", "-s", "-q", "-p", "no:cacheprovider", "test_gen.py"],
                         cwd=root, capture_output=True, text=True)
    return res.stdout, "assert 41 == 42"


def build_output(n_lines: int, error_at: float) -> tuple[str, str]:
    """Run a real process that prints a long build log with one error in it."""
    key = "error: undefined reference to `crc32_update'"
    prog = (f"import sys\nfor i in range({n_lines}):\n"
            f"    print(f'[{{i}}/{n_lines}] CC src/unit_{{i}}.o')\n"
            f"    if i == {int(error_at * n_lines)}: print({'ld: ' + key!r})\n"
            f"print('make: *** [all] Error 2')\n")
    return subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, check=True).stdout, key
