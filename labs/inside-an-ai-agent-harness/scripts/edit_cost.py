"""Section 8: expected tokens per landed edit, E[m] = m / p, and the break-even file size.

    python labs/inside-an-ai-agent-harness/scripts/edit_cost.py

Files are C10's own source, cut or joined to each size. p values are parameters.
Writes results/edit_cost.txt.
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_lab import C10  # noqa: E402
from harness_lab.edit_cost import FORMATS, clean_cost, land_edit  # noqa: E402
from harness_lab.report import Report  # noqa: E402

SIZES = (10, 25, 50, 100, 200, 400, 800)
P = {"whole file": 0.98, "search/replace": 0.85, "unified diff": 0.7}
REMAINING_STEPS = 20


def corpus_lines() -> list[str]:
    out: list[str] = []
    for p in sorted((C10 / "harness").rglob("*.py")):
        out += p.read_text().splitlines()
    return out


def file_of(lines: list[str], n: int, rng: random.Random) -> tuple[str, int]:
    """A file of n lines and a line to edit that is unique and not blank."""
    start = rng.randrange(0, len(lines) - n)
    body = lines[start:start + n]
    text = "\n".join(body) + "\n"
    for _ in range(1000):
        i = rng.randrange(1, n - 1) if n > 2 else 0
        window = body[max(0, i - 1): i + 2]
        if body[i].strip() and body.count(body[i]) == 1 and "\n".join(window) in text and \
                text.count("\n".join(window)) == 1 and (i + 7 >= n or body[i + 7] != body[i]):
            return text, i
    raise RuntimeError("no editable line")


def main() -> None:
    rep = Report("edit_cost.txt")
    rng = random.Random(0)
    lines = corpus_lines()
    rep(f"p per attempt (schematic, as in the post): {P}")
    rep(f"Each size: 400 trials, a fresh file and line each time. Context cost = failed attempts and error "
        f"messages, times {REMAINING_STEPS} later steps.")
    rep()
    rep(f"{'lines':>5} {'format':>15} {'m (clean)':>10} {'m / p':>8} {'measured E[m]':>14} {'mean tries':>11} "
        f"{'1/p':>5} {'context cost':>13}")
    means: dict[tuple[int, str], float] = {}
    for n in SIZES:
        for fmt in FORMATS:
            ms, outs, tries, ctx = [], [], [], []
            for _ in range(400):
                text, i = file_of(lines, n, rng)
                ms.append(clean_cost(fmt, text, i))
                o = land_edit(fmt, text, i, P[fmt], rng)
                outs.append(o.output_tokens)
                tries.append(o.attempts)
                ctx.append(o.context_tokens * REMAINING_STEPS)
            m = statistics.mean(ms)
            means[(n, fmt)] = statistics.mean(outs)
            rep(f"{n:>5} {fmt:>15} {m:>10.0f} {m / P[fmt]:>8.0f} {statistics.mean(outs):>14.0f} "
                f"{statistics.mean(tries):>11.2f} {1 / P[fmt]:>5.2f} {statistics.mean(ctx):>13.0f}")
    rep()
    rep("Cheapest format per file size (measured output tokens per landed edit):")
    for n in SIZES:
        best = min(FORMATS, key=lambda f: means[(n, f)])
        rep(f"  {n:>4} lines: {best}  (" + ", ".join(f"{f} {means[(n, f)]:.0f}" for f in FORMATS) + ")")
    rep()
    rep("Break-even against whole-file rewrite, from the formula |F|* = m_patch * p_whole / p_patch (in tokens):")
    for fmt in ("search/replace", "unified diff"):
        m_patch = statistics.mean(clean_cost(fmt, *file_of(lines, 200, rng)) for _ in range(200))
        rep(f"  {fmt:>15}: m = {m_patch:.0f} tokens -> whole file is cheaper below about "
            f"{m_patch * P['whole file'] / P[fmt]:.0f} tokens of file")
    rep.save()


if __name__ == "__main__":
    main()
