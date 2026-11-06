"""GSM8K questions and answer scoring. Answers are compared as normalised number strings."""

from __future__ import annotations

import random
import re

PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalise(x: str | None) -> str | None:
    if x is None:
        return None
    s = x.strip().replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").rstrip(".")
    m = _NUM.search(s)
    if not m:
        return None
    s = m.group(0).replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.lstrip("+") or None


def gold_answer(solution: str) -> str:
    """GSM8K solutions end with '#### <number>'."""
    return normalise(solution.split("####")[-1])


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, with nested braces handled."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    start = j
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[start : j - 1] if depth == 0 else None


def extract_answer(text: str) -> str | None:
    """The boxed answer if there is one, else the last number in the text."""
    boxed = last_boxed(text)
    if boxed is not None:
        return normalise(boxed)
    nums = _NUM.findall(text)
    return normalise(nums[-1]) if nums else None


def load_questions(n: int, seed: int = 0, split: str = "test") -> list[dict]:
    """n questions sampled with a fixed seed. Needs the dataset (network or local cache)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    idx = sorted(random.Random(seed).sample(range(len(ds)), n))
    return [{"idx": i, "question": ds[i]["question"], "gold": gold_answer(ds[i]["answer"])} for i in idx]
