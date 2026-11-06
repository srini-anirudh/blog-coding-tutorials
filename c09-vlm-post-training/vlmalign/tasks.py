"""Questions about a scene, with exact answers.

Each question carries a sentence template so that a chosen answer and a
rejected answer differ in exactly one slot ("The circle is {red|blue}.").
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .scenes import Scene

KINDS = ("color", "count", "object", "ocr", "spatial")
SHORT_SUFFIX = {
    "color": " Answer with one word.",
    "count": " Answer with a number.",
    "object": " Answer with one word.",
    "ocr": " Answer with one word.",
    "spatial": " Answer with left or right.",
}


@dataclass(frozen=True)
class Question:
    kind: str
    prompt: str    # the question shown to the model
    answer: str    # canonical short answer: "red", "3", "circle", "CAT", "left"
    template: str  # sentence with one slot: "The circle is {}."
    slot: str      # what fills the slot, used by the negative generators

    def sentence(self, value: str | None = None) -> str:
        return self.template.format(self.answer if value is None else value)

    def short_prompt(self) -> str:
        return self.prompt + SHORT_SUFFIX[self.kind]


def ask(scene: Scene, kind: str, rng: random.Random) -> Question | None:
    """Return one question of `kind` about `scene`, or None if the scene cannot support it."""
    if kind == "color":
        unique = [o for o in scene.objects if scene.count(o.shape) == 1]
        if not unique:
            return None
        o = rng.choice(unique)
        return Question(kind, f"What color is the {o.shape}?", o.color, f"The {o.shape} is {{}}.", "color")
    if kind == "count":
        shape = rng.choice(sorted({o.shape for o in scene.objects}))
        n = scene.count(shape)
        return Question(kind, f"How many {shape}s are there?", str(n), f"The number of {shape}s is {{}}.", "count")
    if kind == "object":
        unique = [o for o in scene.objects if len(scene.with_color(o.color)) == 1]
        if not unique:
            return None
        o = rng.choice(unique)
        return Question(kind, f"What shape is the {o.color} object?", o.shape, f"The {o.color} object is a {{}}.", "shape")
    if kind == "ocr":
        if not scene.word:
            return None
        return Question(kind, "What word is written in the image?", scene.word, "The word is {}.", "word")
    if kind == "spatial":
        pairs = [(a, b) for a in scene.objects for b in scene.objects
                 if a is not b and abs(a.center[0] - b.center[0]) > 60]
        if not pairs:
            return None
        a, b = rng.choice(pairs)
        side = "left" if a.center[0] < b.center[0] else "right"
        return Question(kind, f"Is the {a.name} to the left or to the right of the {b.name}?", side,
                        f"The {a.name} is to the {{}} of the {b.name}.", "side")
    raise ValueError(f"unknown kind {kind!r}")
