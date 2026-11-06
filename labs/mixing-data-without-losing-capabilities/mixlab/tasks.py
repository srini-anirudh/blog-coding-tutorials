"""Small synthetic "capabilities" with exact answers.

Each task maps a short prompt to a short answer, written in one shared character vocabulary. The tasks stand in
for the post's maths, code and chat sources. They are cheap to generate and every label is exact, so a drop in a
score is a real drop and not a grading artefact.

    add   "37+45="  -> "082"      arithmetic ("maths")
    sub   "45-37="  -> "08"       arithmetic that shares the digit machinery with add
    rev   "R3745="  -> "5473"     string manipulation ("code")
    sort  "S3745="  -> "3457"     string manipulation that needs comparisons
    copy  "C3745="  -> "3745"     easy, saturates fast ("chat")
    noise "N3745="  -> 4 random digits, a fresh draw each time: the loss can never fall below 4 * ln(10)
    long  "L3745="  -> "3745" repeated, a long answer for the token-weighting demo

Answers end with "\n", which is also supervised. `style` adds a response preamble (">" before the answer), which
is how the spurious-forgetting demo changes an output convention without changing the skill.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

VOCAB = list("0123456789+-=RSCNL>\n") + ["<pad>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD = STOI["<pad>"]
IGNORE = -100

TASKS = ("add", "sub", "rev", "sort", "copy", "noise", "long")


def encode(text: str) -> list[int]:
    return [STOI[c] for c in text]


def decode(ids) -> str:
    return "".join(VOCAB[int(i)] for i in ids if int(i) != PAD)


def make_example(rng: random.Random, task: str, digits: int = 4, long_repeats: int = 3) -> tuple[str, str]:
    """One (prompt, answer) pair. The answer does not include the end-of-answer newline."""
    if task in ("add", "sub"):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if task == "add":
            return f"{a}+{b}=", f"{a + b:03d}"
        a, b = max(a, b), min(a, b)
        return f"{a}-{b}=", f"{a - b:02d}"
    d = "".join(rng.choice("0123456789") for _ in range(digits))
    if task == "rev":
        return f"R{d}=", d[::-1]
    if task == "sort":
        return f"S{d}=", "".join(sorted(d))
    if task == "copy":
        return f"C{d}=", d
    if task == "noise":
        return f"N{d}=", "".join(rng.choice("0123456789") for _ in range(digits))
    if task == "long":
        return f"L{d}=", d * long_repeats
    raise ValueError(f"unknown task {task!r}")


@dataclass
class Pool:
    """A fixed set of tokenised examples from one task.

    input_ids[k] is prompt + answer + "\n", right-padded. labels[k] is -100 except on the answer and the newline,
    already shifted so that labels[k, t] is the target for position t.
    """

    task: str
    input_ids: torch.Tensor  # (N, T)
    labels: torch.Tensor  # (N, T)
    style: str = ""

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    @property
    def answer_tokens(self) -> float:
        """Mean number of supervised tokens per example (the post's mean supervised length)."""
        return float((self.labels != IGNORE).sum(1).float().mean())


def build_pool(task: str, n: int, seed: int, style: str = "", seq_len: int = 20, **kw) -> Pool:
    rng = random.Random(f"{task}-{seed}")
    ids = torch.full((n, seq_len), PAD, dtype=torch.long)
    labels = torch.full((n, seq_len), IGNORE, dtype=torch.long)
    for k in range(n):
        prompt, answer = make_example(rng, task, **kw)
        full = encode(prompt + style + answer + "\n")
        if len(full) > seq_len:
            raise ValueError(f"example longer than seq_len={seq_len}")
        ids[k, : len(full)] = torch.tensor(full)
        # Position t predicts token t + 1. Supervise from the last prompt token up to the token before "\n".
        p = len(prompt)
        labels[k, p - 1 : len(full) - 1] = torch.tensor(full[p:])
    return Pool(task, ids, labels, style)


def prompt_and_answer(pool: Pool, k: int) -> tuple[list[int], list[int]]:
    """Split example k back into its prompt ids and its supervised ids (answer + newline)."""
    lab = pool.labels[k]
    pos = (lab != IGNORE).nonzero().flatten()
    start = int(pos[0]) + 1
    return pool.input_ids[k, :start].tolist(), lab[pos].tolist()
