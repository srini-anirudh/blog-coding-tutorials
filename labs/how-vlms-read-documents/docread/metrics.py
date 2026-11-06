"""String metrics for document reading: edit distance, ANLS, the OmniDocBench overall score, unit tests.

Section "How document reading is measured" of the post.
"""

from __future__ import annotations

from typing import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Fewest insertions, deletions and substitutions that turn `a` into `b`. Works on strings and lists."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def normalized_edit_distance(a: Sequence, b: Sequence) -> float:
    """Edit distance divided by the longer length, so 0 is identical and 1 is nothing in common."""
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def anls_single(prediction: str, golds: Sequence[str], threshold: float = 0.5) -> float:
    """ST-VQA's score for one question: best 1 - NL over the gold answers, and 0 if NL is not below the threshold."""
    best = 0.0
    for gold in golds:
        nl = normalized_edit_distance(prediction.strip().lower(), gold.strip().lower())
        best = max(best, 1.0 - nl if nl < threshold else 0.0)
    return best


def anls(predictions: Sequence[str], golds: Sequence[Sequence[str]], threshold: float = 0.5) -> float:
    """Average ANLS over questions."""
    return sum(anls_single(p, g, threshold) for p, g in zip(predictions, golds)) / len(predictions)


def omnidocbench_overall(text_edit: float, table_teds: float, formula_cdm: float) -> float:
    """OmniDocBench v1.5: [(1 - TextEdit) x 100 + Table TEDS + Formula CDM] / 3. Reading order is not in it."""
    return ((1.0 - text_edit) * 100.0 + table_teds + formula_cdm) / 3.0


# olmOCR-Bench style binary unit tests. Each takes the model output and returns pass or fail.

def check_present(output: str, phrase: str) -> bool:
    return " ".join(phrase.split()) in " ".join(output.split())


def check_absent(output: str, phrase: str) -> bool:
    return not check_present(output, phrase)


def check_order(output: str, first: str, second: str) -> bool:
    text = " ".join(output.split())
    i, j = text.find(" ".join(first.split())), text.find(" ".join(second.split()))
    return i >= 0 and j >= 0 and i < j


def longest_repeat_run(output: str) -> int:
    """How many times in a row the most repeated line appears. A decoder stuck in a loop gives a large number."""
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    best = run = 1 if lines else 0
    for prev, cur in zip(lines, lines[1:]):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    return best


def check_no_repetition(output: str, max_run: int = 3) -> bool:
    return longest_repeat_run(output) <= max_run
