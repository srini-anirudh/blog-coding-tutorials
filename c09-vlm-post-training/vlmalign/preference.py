"""Multimodal preference pairs: (image, prompt, chosen, rejected).

The chosen answer states the true fact. The rejected answer is a programmatic
hallucination: the same sentence with one slot changed. Five kinds:

    color    wrong colour        "The circle is red."      vs "The circle is blue."
    count    wrong count         "The number of squares is 2." vs "... is 3."
    object   wrong object        "The red object is a circle." vs "... is a square."
    ocr      false OCR           "The word is CAT."        vs "The word is CUP."
    spatial  wrong relation      "... to the left of ..."  vs "... to the right of ..."

Two negative samplers share everything except how the wrong value is drawn:

    balanced  the wrong value comes from the same distribution as true values,
              so the pair carries no text-only hint about which side is chosen.
    biased    the wrong value comes from a different distribution (never-drawn
              colours, always over-counting, misspelled words). A model can
              learn these preferences without ever looking at the image.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image

from .scenes import OFF_PALETTE, PALETTE, SHAPES, WORDS, Scene, render, sample_scene
from .tasks import KINDS, Question, ask

# The value set for each slot. Balanced sampling draws true AND wrong values
# uniformly from these, which makes (chosen, rejected) exchangeable.
VALUES: dict[str, list[str]] = {
    "color": list(PALETTE),
    "count": ["1", "2", "3"],
    "object": list(SHAPES),
    "ocr": list(WORDS),
    "spatial": ["left", "right"],
}
_OTHER_SIDE = {"left": "right", "right": "left"}


def wrong_value(q: Question, rng: random.Random, mode: str = "balanced") -> str:
    """A value for the slot that is false for this image."""
    if q.kind == "spatial":
        return _OTHER_SIDE[q.answer]  # only two values: flipping is already symmetric
    if mode == "balanced":
        return rng.choice([v for v in VALUES[q.kind] if v != q.answer])
    if mode == "biased":
        if q.kind == "color":
            return rng.choice(OFF_PALETTE)      # a colour no image ever contains
        if q.kind == "count":
            return str(int(q.answer) + 1)       # always one too many
        if q.kind == "object":
            return rng.choice(["star", "hexagon"])  # a shape no image ever contains
        if q.kind == "ocr":
            i = rng.randrange(len(q.answer))    # a misspelling, never a real word
            return q.answer[:i] + rng.choice("QXZJ") + q.answer[i + 1:]
    raise ValueError(f"unknown mode {mode!r}")


@dataclass
class PreferencePair:
    scene: Scene
    question: Question
    chosen: str
    rejected: str
    mode: str

    @property
    def kind(self) -> str:
        return self.question.kind

    @property
    def prompt(self) -> str:
        return self.question.prompt

    def image(self) -> Image.Image:
        return render(self.scene)


def build_pairs(n: int, mode: str = "balanced", seed: int = 0, kinds: tuple[str, ...] = KINDS) -> list[PreferencePair]:
    """n pairs, kinds in round-robin, true answers uniform over VALUES[kind].

    We pick the target true value first and then search scenes until one has
    that answer. Without this stratification, true counts skew towards 1 and 2,
    and "3" would mostly appear on the rejected side: a text-only shortcut.
    """
    rng = random.Random(seed)
    pairs: list[PreferencePair] = []
    scene_seed = seed * 1_000_003
    for i in range(n):
        kind = kinds[i % len(kinds)]
        target = rng.choice(VALUES[kind])
        while True:
            scene_seed += 1
            scene = sample_scene(scene_seed)
            q = ask(scene, kind, rng)
            if q is not None and q.answer == target:
                break
        wrong = wrong_value(q, rng, mode)
        pairs.append(PreferencePair(scene, q, q.sentence(), q.sentence(wrong), mode))
    return pairs
