"""Public instruction data (no login) and fixed, seeded splits."""

from __future__ import annotations

import random

from datasets import load_dataset

ALPACA = "yahma/alpaca-cleaned"          # 51,760 teacher-written pairs (text-davinci-003), cleaned
DOLLY = "databricks/databricks-dolly-15k"  # 15,011 human-written pairs, 8 categories


def user_text(instruction: str, context: str = "") -> str:
    return instruction.strip() + (("\n\n" + context.strip()) if context and context.strip() else "")


def load_alpaca() -> list[dict]:
    d = load_dataset(ALPACA, split="train")
    return [{"prompt": user_text(r["instruction"], r["input"]), "response": r["output"].strip(),
             "instruction": r["instruction"].strip(), "context": r["input"].strip(), "source": "alpaca"} for r in d
            if r["output"].strip()]


def load_dolly() -> list[dict]:
    d = load_dataset(DOLLY, split="train")
    return [{"prompt": user_text(r["instruction"], r["context"]), "response": r["response"].strip(),
             "instruction": r["instruction"].strip(), "context": r["context"].strip(),
             "category": r["category"], "source": "dolly"} for r in d if r["response"].strip()]


def to_messages(ex: dict) -> list[dict]:
    return [{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": ex["response"]}]


def split(items: list, n_train: int, n_val: int, seed: int = 0) -> tuple[list, list]:
    """Seeded shuffle, then disjoint train and validation. The validation set is the same for every n_train
    at a given seed, because it is taken from the front of the shuffled list."""
    r = random.Random(seed)
    idx = list(range(len(items)))
    r.shuffle(idx)
    val = [items[i] for i in idx[:n_val]]
    train = [items[i] for i in idx[n_val:n_val + n_train]]
    return train, val
