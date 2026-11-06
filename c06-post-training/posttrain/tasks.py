"""Synthetic arithmetic conversations with an exact, checkable answer.

One task family carries the whole tutorial. SFT teaches the answer format, DPO prefers correct answers over
near-miss wrong ones, and GRPO optimises the exact reward `1 if answer == truth else 0`. Because we generate the
data, every label and every reward is exact.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

SYSTEM_PROMPT = "You are a careful calculator. Work step by step, then finish with 'The answer is N.'"
KINDS = ("add", "sub", "chain", "mul")
ANSWER_RE = re.compile(r"The answer is (-?\d+)\.")


@dataclass
class Problem:
    question: str
    steps: list[str]
    answer: int
    kind: str
    follow_up: tuple[str, list[str], int] | None = field(default=None)

    def solution(self) -> str:
        return " ".join(self.steps) + f" The answer is {self.answer}."


def _op(a: int, sign: str, b: int) -> tuple[str, int]:
    value = a + b if sign == "+" else a - b if sign == "-" else a * b
    return f"{a} {sign} {b} = {value}.", value


def make_problem(rng: random.Random, kind: str, digits: int = 2) -> Problem:
    """One problem of the given kind. `digits` sets the operand size for add/sub/chain."""
    lo, hi = 10 ** (digits - 1), 10**digits - 1
    a, b, c = rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi)
    if kind == "add":
        step, ans = _op(a, "+", b)
        return Problem(f"What is {a} + {b}?", [step], ans, kind)
    if kind == "sub":
        a, b = max(a, b), min(a, b)
        step, ans = _op(a, "-", b)
        return Problem(f"What is {a} - {b}?", [step], ans, kind)
    if kind == "chain":
        s1, v1 = _op(a, "+", b)
        s2, ans = _op(v1, "-", c)
        return Problem(f"What is {a} + {b} - {c}?", [s1, s2], ans, kind)
    if kind == "mul":
        a = rng.randint(2, 9)
        step, ans = _op(a, "*", b)
        return Problem(f"What is {a} * {b}?", [step], ans, kind)
    raise ValueError(f"unknown kind {kind!r}")


def add_follow_up(rng: random.Random, p: Problem) -> Problem:
    """Attach a second user turn ("Now add 7 to that.") so conversations have two assistant turns."""
    d = rng.randint(2, 30)
    sign = rng.choice("+-")
    step, value = _op(p.answer, sign, d)
    text = f"Now add {d} to that." if sign == "+" else f"Now subtract {d} from that."
    p.follow_up = (text, [step], value)
    return p


def to_messages(p: Problem, with_system: bool = True, with_answer: bool = True) -> list[dict[str, str]]:
    """Render a problem as chat messages. Without the answer, the list ends on the user turn (a prompt)."""
    msgs: list[dict[str, str]] = []
    if with_system:
        msgs.append({"role": "system", "content": SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": p.question})
    if not with_answer:
        return msgs
    msgs.append({"role": "assistant", "content": p.solution()})
    if p.follow_up is not None:
        text, steps, value = p.follow_up
        msgs.append({"role": "user", "content": text})
        msgs.append({"role": "assistant", "content": " ".join(steps) + f" The answer is {value}."})
    return msgs


def sample_problems(
    n: int, seed: int, kinds: tuple[str, ...] = KINDS, follow_up_frac: float = 0.0, digits: int = 2
) -> list[Problem]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        p = make_problem(rng, kinds[i % len(kinds)], digits)
        if rng.random() < follow_up_frac:
            add_follow_up(rng, p)
        out.append(p)
    return out


def split_problems(n_train: int, n_test: int, seed: int = 0, **kw) -> tuple[list[Problem], list[Problem]]:
    """Train and test sets with no question in common."""
    test = sample_problems(n_test, seed + 10_000, **{k: v for k, v in kw.items() if k != "follow_up_frac"})
    held_out = {p.question for p in test}
    train = [p for p in sample_problems(int(n_train * 1.2), seed, **kw) if p.question not in held_out][:n_train]
    return train, test


def extract_answer(text: str) -> int | None:
    """The first 'The answer is N.' in a completion, or None if the format is broken."""
    m = ANSWER_RE.search(text)
    return int(m.group(1)) if m else None


def exact_reward(completion: str, truth: int) -> float:
    """The verifiable reward: 1.0 for the exact right number in the right format, else 0.0."""
    return 1.0 if extract_answer(completion) == truth else 0.0


def near_miss(rng: random.Random, p: Problem) -> str:
    """A plausible wrong solution: the same steps with one result off by a carry-sized error."""
    delta = rng.choice([-10, -1, 1, 10, 2, -2])
    wrong = p.answer + delta
    steps = list(p.steps)
    lhs = steps[-1].rsplit("=", 1)[0]
    steps[-1] = f"{lhs}= {wrong}."
    return " ".join(steps) + f" The answer is {wrong}."
