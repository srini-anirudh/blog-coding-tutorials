"""The capability vector: one fixed, held-out suite per skill. Every suite draws from its own seed."""

from __future__ import annotations

import random

from .model import TinyVLM
from .scenes import OBJECTS, Scene, recolour, sample_scene
from .tasks import FORMAT_PROMPT, _question, reason_answer, reason_scene, text_reverse
from .train import generate

EMPTY = Scene(())


def shown(scene: Scene, image: str, rng: random.Random, obj: str | None = None) -> Scene:
    """The scene the model is shown. real: the true one. blank: no objects (image removed).
    swap: an unrelated scene that still holds the object the question names, so the question stays valid."""
    if image == "real":
        return scene
    if image == "blank":
        return EMPTY
    return sample_scene(rng, require=obj)


def _answer(scene: Scene, kind: str, it) -> str:
    return str(len(scene.items)) if kind == "count" else (it.colour if kind == "color" else it.where)


def _questions(n: int, seed: int):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = sample_scene(rng)
        kind, q, it = _question(s, rng)
        out.append((s, q, _answer(s, kind, it), kind))
    return out


def eval_vqa(model: TinyVLM, n: int = 300, fmt: bool = True, image: str = "real", seed: int = 101) -> dict:
    """Short-answer accuracy: exact match. fmt adds the format prompt, as at training time."""
    qs = _questions(n, seed)
    suffix = f" {FORMAT_PROMPT}" if fmt else ""
    rng = random.Random(seed + 1)
    obj = [next((o for o in OBJECTS if f" {o} " in f" {q} "), None) for _, q, _, _ in qs]
    outs = generate(model, [([("user", f"<image> {q}{suffix}")], shown(s, image, rng, o))
                            for (s, q, _, _), o in zip(qs, obj)])
    acc = sum(o == a for o, (_, _, a, _) in zip(outs, qs)) / n
    return {"vqa_exact": round(acc, 3)}


def eval_chat(model: TinyVLM, n: int = 300, seed: int = 202) -> dict:
    """Unmarked questions, as a user would ask them. Long form = at least 6 words.
    Correct = the right colour, position or count appears in the answer."""
    qs = _questions(n, seed)
    outs = generate(model, [([("user", f"<image> {q}")], s) for s, q, _, _ in qs])
    lens = [len(o.split()) for o in outs]
    correct = sum(f" {a} " in f" {o} " for o, (_, _, a, _) in zip(outs, qs))
    return {"chat_long_rate": round(sum(l >= 6 for l in lens) / n, 3),
            "chat_mean_words": round(sum(lens) / n, 2), "chat_correct": round(correct / n, 3)}


def eval_text(model: TinyVLM, n: int = 300, seed: int = 303) -> dict:
    """Text-only suite: reverse a word list. Exact match."""
    rng = random.Random(seed)
    items = [text_reverse(rng) for _ in range(n)]
    outs = generate(model, [([t[0]], None) for t in items])
    return {"text_exact": round(sum(o == t[1][1] for o, t in zip(outs, items)) / n, 3)}


def mentioned(text: str) -> set[str]:
    return {o for o in OBJECTS if f" {o} " in f" {text} "}


def eval_caption(model: TinyVLM, n: int = 300, seed: int = 404) -> dict:
    """Captions on scenes that hold a plate but no fork: the case a biased teacher gets wrong.
    chair_i   = mentioned objects not in the scene / all mentioned objects (CHAIR, instance level)
    fork_rate = share of captions that mention the absent fork
    recall    = share of present objects that the caption mentions"""
    rng = random.Random(seed)
    scenes = [sample_scene(rng, 1, 3, require="plate", forbid="fork") for _ in range(n)]
    outs = generate(model, [([("user", "<image> describe the image .")], s) for s in scenes])
    ment = [mentioned(o) for o in outs]
    wrong = sum(len(m - {it.obj for it in s.items}) for m, s in zip(ment, scenes))
    total = max(1, sum(len(m) for m in ment))
    rec = sum(len(m & {it.obj for it in s.items}) for m, s in zip(ment, scenes)) / sum(len(s.items) for s in scenes)
    return {"chair_i": round(wrong / total, 3), "fork_rate": round(sum("fork" in m for m in ment) / n, 3),
            "recall": round(rec, 3), "sample": outs[0]}


def eval_exist(model: TinyVLM, n: int = 300, seed: int = 505) -> dict:
    """POPE-style probe on the same plate-without-fork scenes: how often does "is there a fork ?" get yes."""
    rng = random.Random(seed)
    scenes = [sample_scene(rng, 1, 3, require="plate", forbid="fork") for _ in range(n)]
    outs = generate(model, [([("user", "<image> is there a fork ?")], s) for s in scenes])
    return {"fork_yes_rate": round(sum(o == "yes" for o in outs) / n, 3)}


def eval_reason(model: TinyVLM, n: int = 300, seed: int = 606, image: str = "real") -> dict:
    """Answer accuracy, and the counterfactual check: repaint one blue object red and ask again.
    A rationale that reads the image must change its numbers to match the edited scene."""
    rng = random.Random(seed)
    scenes = []
    while len(scenes) < n:
        s = reason_scene(rng)
        if s.count("blue") > 0:
            scenes.append(s)
    q = [("user", "<image> how many more red than blue objects ?")]
    edited = [recolour(s, "blue", "red") for s in scenes]
    if image != "real":  # the same unrelated scene before and after the edit
        alt = [shown(s, image, rng) for s in scenes]
        scenes_in, edited_in = alt, alt
    else:
        scenes_in, edited_in = scenes, edited
    outs = generate(model, [(q, s) for s in scenes_in])
    outs_cf = generate(model, [(q, s) for s in edited_in])
    acc = sum(o == reason_answer(s) for o, s in zip(outs, scenes)) / n
    cf = sum(o == reason_answer(s) for o, s in zip(outs_cf, edited)) / n
    unchanged = sum(a == b for a, b in zip(outs, outs_cf)) / n
    return {"reason_exact": round(acc, 3), "counterfactual_exact": round(cf, 3),
            "rationale_unchanged": round(unchanged, 3)}


def eval_fork_colour(model: TinyVLM, n: int = 200, image: str = "real", seed: int = 707) -> dict:
    """A question the image is not needed for: forks are always grey in this world."""
    rng = random.Random(seed)
    scenes = [sample_scene(rng, 1, 4, require="fork") for _ in range(n)]
    outs = generate(model, [([("user", f"<image> what color is the fork ? {FORMAT_PROMPT}")],
                              shown(s, image, rng, "fork")) for s in scenes])
    return {"fork_colour_exact": round(sum(o == "grey" for o in outs) / n, 3)}
