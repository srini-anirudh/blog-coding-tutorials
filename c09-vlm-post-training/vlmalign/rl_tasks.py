"""RL tasks: a prompt per scene and a reward computed from the scene itself."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .rewards import caption_f1, caption_recall, contains_match, exact_match, mentioned
from .scenes import PALETTE, Scene, sample_scene
from .tasks import ask

# Colour words the caption rewards look for. Includes colours never drawn, so
# naming them counts as a hallucination.
COLOR_VOCAB = list(PALETTE) + ["purple", "orange", "pink", "black", "white", "brown", "gray", "grey"]
CAPTION_PROMPT = "List the colors of the shapes in the image."


@dataclass
class RLExample:
    scene: Scene
    prompt: str
    answer: str = ""                        # short exact answer (count task)
    truth: set[str] = field(default_factory=set)  # true colour set (caption task)


def count_examples(n: int, seed: int) -> list[RLExample]:
    rng = random.Random(seed)
    out, s = [], seed * 104_729
    while len(out) < n:
        s += 1
        scene = sample_scene(s, with_word=False)
        q = ask(scene, "count", rng)
        out.append(RLExample(scene, q.short_prompt(), answer=q.answer))
    return out


def caption_examples(n: int, seed: int) -> list[RLExample]:
    out = []
    for i in range(n):
        scene = sample_scene(seed * 104_729 + i, min_objects=2, max_objects=4, with_word=False)
        out.append(RLExample(scene, CAPTION_PROMPT, truth={o.color for o in scene.objects}))
    return out


Reward = Callable[[str, RLExample], float]
REWARDS: dict[str, Reward] = {
    "exact": lambda text, ex: exact_match(text, ex.answer),
    "contains": lambda text, ex: contains_match(text, ex.answer),          # weak
    "color_recall": lambda text, ex: caption_recall(text, ex.truth, COLOR_VOCAB),  # weak
    "color_f1": lambda text, ex: caption_f1(text, ex.truth, COLOR_VOCAB),
}
TASKS = {"count": count_examples, "caption": caption_examples}


def caption_stats(text: str, ex: RLExample) -> dict[str, float]:
    """Monitors for the caption task: how many colours named, how many were not in the image."""
    said = mentioned(text, COLOR_VOCAB)
    return {"n_colors": len(said), "hallucinated": len(said - ex.truth),
            "f1": caption_f1(text, ex.truth, COLOR_VOCAB), "recall": caption_recall(text, ex.truth, COLOR_VOCAB)}
