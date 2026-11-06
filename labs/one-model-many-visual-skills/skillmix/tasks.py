"""Instruction sources. Each builder returns (turns, scene); scene is None for text-only data.

    vqa      academic short answers: "red", "3", "top left". With fmt=True the question carries
             the LLaVA-1.5 style format prompt "answer with a word or phrase ."
    chat     the same questions answered in full sentences, as an assistant would
    caption  "describe the image ." -> one sentence per object, in reading order. A biased teacher
             adds "there is a grey fork ..." at the end when a plate has no fork (the co-occurrence)
    exist    "is there a fork ?" -> yes / no, half of them negative (LRV-style negatives)
    reason   "how many more red than blue objects ?" with a short rationale before the answer
    text     text-only: reverse a list of words. This is the language model's own skill
"""

from __future__ import annotations

import random

from .scenes import GRID, OBJECTS, POSITIONS, Scene, sample_scene

FORMAT_PROMPT = "answer with a word or phrase ."
TEXT_WORDS = ("apple tree river stone cloud music paper glass window chair garden light "
              "red green blue yellow plate cup ball book").split()

Turns = list[tuple[str, str]]


def _question(scene: Scene, rng: random.Random) -> tuple[str, str, object]:
    kind = rng.choice(["color", "count", "where"])
    if kind == "count":
        return kind, "how many objects are there ?", None
    it = rng.choice(scene.items)
    q = f"what color is the {it.obj} ?" if kind == "color" else f"where is the {it.obj} ?"
    return kind, q, it


def vqa(scene: Scene, rng: random.Random, fmt: bool = False) -> Turns:
    kind, q, it = _question(scene, rng)
    a = str(len(scene.items)) if kind == "count" else (it.colour if kind == "color" else it.where)
    return [("user", f"<image> {q}" + (f" {FORMAT_PROMPT}" if fmt else "")), ("assistant", a)]


def chat(scene: Scene, rng: random.Random) -> Turns:
    kind, q, it = _question(scene, rng)
    if kind == "count":
        n = len(scene.items)
        a = f"there are {n} objects in the image . the first one is a {scene.items[0].colour} {scene.items[0].obj} ."
    elif kind == "color":
        a = f"the {it.obj} is {it.colour} . it is at the {it.where} of the image ."
    else:
        a = f"the {it.obj} is at the {it.where} of the image . it is {it.colour} ."
    return [("user", f"<image> {q}"), ("assistant", a)]


def caption_text(scene: Scene, rng: random.Random, teacher_bias: float = 0.0) -> str:
    sents = [f"there is a {it.colour} {it.obj} at the {it.where} ." for it in scene.items]
    if teacher_bias and scene.has("plate") and not scene.has("fork") and rng.random() < teacher_bias:
        free = [c for c in range(GRID * GRID) if all(it.cell != c for it in scene.items)]
        sents.append(f"there is a grey fork at the {POSITIONS[rng.choice(free)]} .")  # late, as LURE observed
    return " ".join(sents)


def verify_caption(text: str, scene: Scene) -> str:
    """Object-level check: drop every sentence that names an object the scene does not contain.
    Here the check reads the ground-truth scene; a real pipeline would use a detector or a grounding model."""
    sents = [s.strip() + " ." for s in text.split(" .") if s.strip()]
    keep = [s for s in sents if all(scene.has(o) for o in OBJECTS if f" {o} " in f" {s} ")]
    return " ".join(keep)


def caption(scene: Scene, rng: random.Random, teacher_bias: float = 0.0, verify: bool = False) -> Turns:
    text = caption_text(scene, rng, teacher_bias)
    if verify:
        text = verify_caption(text, scene)
    return [("user", "<image> describe the image ."), ("assistant", text)]


def exist(scene: Scene, rng: random.Random) -> Turns:
    obj = rng.choice(OBJECTS)
    return [("user", f"<image> is there a {obj} ?"), ("assistant", "yes" if scene.has(obj) else "no")]


def reason_answer(scene: Scene) -> str:
    r, b = scene.count("red"), scene.count("blue")
    return f"red {r} . blue {b} . {r} minus {b} is {r - b} ."


def reason(scene: Scene, rng: random.Random) -> Turns:
    return [("user", "<image> how many more red than blue objects ?"), ("assistant", reason_answer(scene))]


def text_reverse(rng: random.Random) -> Turns:
    words = [rng.choice(TEXT_WORDS) for _ in range(rng.randint(3, 6))]
    return [("user", "reverse : " + " ".join(words)), ("assistant", " ".join(reversed(words)))]


def reason_scene(rng: random.Random) -> Scene:
    """Scenes for the reasoning task need at least as many red objects as blue ones."""
    while True:
        s = sample_scene(rng, 2, 4)
        if s.count("red") >= s.count("blue") and s.count("red") + s.count("blue") > 0:
            return s


def make_example(source: str, rng: random.Random, fmt: bool = False, teacher_bias: float = 0.0,
                 verify: bool = False) -> tuple[Turns, Scene | None]:
    if source == "text":
        return text_reverse(rng), None
    if source == "reason":
        s = reason_scene(rng)
        return reason(s, rng), s
    s = sample_scene(rng)
    if source == "vqa":
        return vqa(s, rng, fmt), s
    if source == "chat":
        return chat(s, rng), s
    if source == "caption":
        return caption(s, rng, teacher_bias, verify), s
    if source == "exist":
        return exist(s, rng), s
    raise ValueError(source)
