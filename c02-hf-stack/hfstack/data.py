"""A synthetic arithmetic dataset and its Hugging Face Datasets pipeline.

Synthetic data keeps every label exact: the correct answer to "What is 17 + 25?" is known, so SFT targets,
preference pairs and RL rewards can all be computed without a judge model.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, IterableDataset

SYSTEM_PROMPT = "You are a careful calculator. Answer with the result only."


def make_arithmetic_rows(n: int, seed: int = 0, max_operand: int = 99) -> list[dict]:
    """Return `n` rows like {"id": 3, "a": 17, "b": 25, "op": "+", "answer": 42}."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.randint(0, max_operand), rng.randint(0, max_operand)
        op = rng.choice("+-")
        answer = a + b if op == "+" else a - b
        rows.append({"id": i, "a": a, "b": b, "op": op, "answer": answer})
    return rows


def question(row: dict) -> str:
    return f"What is {row['a']} {row['op']} {row['b']}?"


def to_messages(row: dict) -> dict:
    """Conversational SFT format: one `messages` list per example."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question(row)},
            {"role": "assistant", "content": str(row["answer"])},
        ]
    }


def to_preference(row: dict) -> dict:
    """Conversational preference format: a shared prompt, a chosen reply and a rejected reply."""
    wrong = row["answer"] + random.Random(row["id"]).choice([-10, -2, -1, 1, 2, 10])
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question(row)}],
        "chosen": [{"role": "assistant", "content": str(row["answer"])}],
        "rejected": [{"role": "assistant", "content": str(wrong)}],
    }


def to_prompt_only(row: dict) -> dict:
    """RL format: only the prompt. The trainer generates completions; `answer` feeds the reward function."""
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question(row)}],
        "answer": str(row["answer"]),
    }


def write_jsonl(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def arithmetic_dataset_dict(n_train: int = 512, n_test: int = 64, seed: int = 0) -> DatasetDict:
    """A `DatasetDict` with `train` and `test` splits, built in memory (backed by Arrow tables)."""
    rows = make_arithmetic_rows(n_train + n_test, seed=seed)
    return DatasetDict(
        {
            "train": Dataset.from_list(rows[:n_train]),
            "test": Dataset.from_list(rows[n_train:]),
        }
    )


def arithmetic_stream(n: int, seed: int = 0) -> IterableDataset:
    """The same rows as an `IterableDataset`: produced lazily, never materialised as a table."""

    def gen():
        yield from make_arithmetic_rows(n, seed=seed)

    return IterableDataset.from_generator(gen)


def random_token_dataset(n: int, vocab_size: int, min_len: int = 8, max_len: int = 32, seed: int = 0) -> Dataset:
    """Pre-tokenized sequences of varying length, for exact loop-equivalence tests that need no tokenizer.

    Each row has `input_ids` and `labels`; the first quarter of each sequence is masked to -100,
    as a prompt would be, so token counts differ between rows.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        ids = [rng.randint(3, vocab_size - 1) for _ in range(length)]
        prompt_len = length // 4
        rows.append({"input_ids": ids, "labels": [-100] * prompt_len + ids[prompt_len:]})
    return Dataset.from_list(rows)
