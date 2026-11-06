"""A word-level vocabulary, a ChatML-style template, image placeholder expansion and the loss mask.

    <|im_start|> user <image> what color is the cup ? <|im_end|> <|im_start|> assistant red <|im_end|>

The mask rule from the post: assistant tokens get 1, and so does the assistant's <|im_end|>, because
the model must learn to stop. Template tokens, user text and every visual token get 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scenes import COLOURS, OBJECTS, POSITIONS

SPECIALS = ["<pad>", "<|im_start|>", "<|im_end|>", "user", "assistant", "system", "<image>"]
WORDS = sorted(set(
    list(COLOURS) + OBJECTS + " ".join(POSITIONS).split() + [str(i) for i in range(10)]
    + "what color is the how many objects are there where describe image it at answer with a word or phrase "
      "reverse : yes no and , ? . in of more than so red blue minus count one first step by"
      " apple tree river stone cloud music paper glass window chair garden light".split()))
PAD, IM_START, IM_END, USER, ASSISTANT, SYSTEM, IMAGE = range(len(SPECIALS))


class Vocab:
    def __init__(self) -> None:
        self.itos = SPECIALS + [w for w in WORDS if w not in SPECIALS]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids)


VOCAB = Vocab()


@dataclass
class Serialized:
    ids: list[int]
    mask: list[int]   # 1 where the token is a training target
    kinds: list[str]  # "template", "visual", "user" or "assistant", for display


def serialize(turns: list[tuple[str, str]], n_visual: int, add_generation_prompt: bool = False) -> Serialized:
    """Turn [(role, text), ...] into one sequence. Each <image> in user text becomes n_visual tokens."""
    ids, mask, kinds = [], [], []

    def put(tok: int, m: int, kind: str) -> None:
        ids.append(tok)
        mask.append(m)
        kinds.append(kind)

    for role, text in turns:
        put(IM_START, 0, "template")
        put(VOCAB.stoi[role], 0, "template")
        target = 1 if role == "assistant" else 0
        for w in text.split():
            if w == "<image>":
                for _ in range(n_visual):
                    put(IMAGE, 0, "visual")
            else:
                put(VOCAB.stoi[w], target, role if role != "system" else "template")
        put(IM_END, target, "template" if not target else "assistant")
    if add_generation_prompt:
        put(IM_START, 0, "template")
        put(ASSISTANT, 0, "template")
    return Serialized(ids, mask, kinds)


def count_positions(s: Serialized) -> dict[str, int]:
    """How many positions of each kind, and how many carry loss."""
    out = {k: s.kinds.count(k) for k in ("template", "visual", "user", "assistant")}
    out["total"] = len(s.ids)
    out["loss_tokens"] = sum(s.mask)
    return out
