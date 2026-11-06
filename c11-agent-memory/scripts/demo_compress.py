"""Compress one long test log and one long conversation, and check what survives.

    python c11-agent-memory/scripts/demo_compress.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from agentmem import Block, count_tokens, truncate_to_tokens  # noqa: E402
from agentmem.compress import compact_history, edge_order, extract_lines, extractive_summary  # noqa: E402

LOG = "\n".join(
    [f"tests/test_core.py::test_case_{i} PASSED" for i in range(120)]
    + ["tests/test_core.py::test_median_even FAILED", "E   assert 2 == 2.5", "E   median([1, 2, 3, 4]) returned the upper middle"]
    + [f"tests/test_io.py::test_case_{i} PASSED" for i in range(120)]
    + ["=== 1 failed, 240 passed in 3.21s ==="])
TURNS = ["The deploy region for atlas is eu-west-1. It was picked last year after the audit.",
         "Thanks, that helps.", "The design review moved to Thursday.",
         "Heads up: we moved the deploy region for atlas to us-east-2.",
         "I think the owner for atlas might be chen, not sure.", "The owner for atlas is priya.",
         "Forget the database for atlas.", "Let's grab lunch after this."]
BUDGET = 60


def main() -> None:
    print(f"tool log: {count_tokens(LOG)} tokens, budget {BUDGET}")
    print(f"{'strategy':<24}{'tokens':>7}  keeps the assertion?  keeps the summary line?")
    for name, text in (("head", truncate_to_tokens(LOG, BUDGET)), ("tail", truncate_to_tokens(LOG, BUDGET, keep="tail")),
                       ("head and tail", truncate_to_tokens(LOG, BUDGET, keep="both")), ("extract_lines", extract_lines(LOG, BUDGET))):
        print(f"{name:<24}{count_tokens(text):>7}  {'assert 2 == 2.5' in text!s:<20}  {'1 failed' in text}")
    print("\nextract_lines output:\n" + extract_lines(LOG, BUDGET))

    print(f"\nconversation: {sum(count_tokens(t) for t in TURNS)} tokens")
    summary = extractive_summary(TURNS, 45)
    print(f"extractive_summary, 45 tokens ({count_tokens(summary)} used):\n{summary}")

    events = [Block("events", t, value=0.5, order=i, id=f"e{i}") for i, t in enumerate(TURNS)]
    compact = compact_history(events, keep_recent=2, max_summary_tokens=30)
    print(f"\ncompact_history keep_recent=2: {len(events)} blocks -> {len(compact)} blocks")
    print("\n---\n".join(b.text for b in compact))

    ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5), ("d", 0.3), ("e", 0.1)]
    print("\nedge_order by value:", [n for n, _ in edge_order(ranked, key=lambda p: p[1])])


if __name__ == "__main__":
    main()
