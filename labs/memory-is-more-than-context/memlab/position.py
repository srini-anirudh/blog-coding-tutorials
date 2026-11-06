"""Section 5, "Accessible is not usable": move one fact through the context.

This is the key-value retrieval task from Lost in the Middle (Liu et al.), at small scale:
a JSON object of random UUID pairs, then one key, and the model must return its value.
The pair that holds the answer is placed at a chosen position. A "restate" variant adds the
one line that matters just before the question, which is the post's remedy (section 14).
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass

INSTRUCTION = ("Extract the value corresponding to the specified key in the JSON object below. "
               "Reply with the value only.")


@dataclass(frozen=True)
class KVCase:
    prompt: str
    answer: str
    needle: str  # the text of the answering pair, to locate it in the prompt
    position: int
    n_pairs: int


def make_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def kv_case(n_pairs: int, position: int, rng: random.Random, restate: bool = False) -> KVCase:
    """`position` is the 0-based index of the answering pair among `n_pairs`."""
    assert 0 <= position < n_pairs
    pairs = [(make_uuid(rng), make_uuid(rng)) for _ in range(n_pairs)]
    key, value = pairs[position]
    body = ",\n ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in pairs)
    needle = f"{json.dumps(key)}: {json.dumps(value)}"
    prompt = f"{INSTRUCTION}\n\nJSON data:\n{{{body}}}\n\n"
    if restate:
        prompt += f"Relevant pair (restated from the data): {needle}\n\n"
    prompt += f"Key: {json.dumps(key)}\nCorresponding value:"
    return KVCase(prompt, value, needle, position, n_pairs)


def positions_for(n_pairs: int, slots: int = 5) -> list[int]:
    """Evenly spaced positions from first to last, like the paper's x axis."""
    if slots == 1:
        return [0]
    return sorted({round(i * (n_pairs - 1) / (slots - 1)) for i in range(slots)})
