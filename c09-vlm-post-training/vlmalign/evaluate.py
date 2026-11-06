"""Evaluation: answer accuracy, preference accuracy with and without the image."""

from __future__ import annotations

import random
from collections import defaultdict

import torch
from PIL import Image

from .dpo import pair_batch, pair_logprobs
from .model import generate
from .preference import PreferencePair
from .rewards import exact_match
from .scenes import Scene, blank_like, render, sample_scene
from .tasks import KINDS, Question, ask


def first_word(text: str) -> str:
    words = text.strip().split()
    return words[0] if words else ""


def vqa_items(n_per_kind: int, seed: int, kinds: tuple[str, ...] = KINDS) -> list[tuple[Scene, Question]]:
    """A fixed evaluation set: n_per_kind questions per kind on fresh scenes."""
    rng = random.Random(seed)
    items, s = [], seed * 7919
    for kind in kinds:
        got = 0
        while got < n_per_kind:
            s += 1
            scene = sample_scene(s)
            q = ask(scene, kind, rng)
            if q is not None:
                items.append((scene, q))
                got += 1
    return items


@torch.no_grad()
def vqa_accuracy(model, processor, items: list[tuple[Scene, Question]],
                 image_fn=render) -> dict[str, float]:
    """Greedy short answers, scored by exact match on the first word."""
    model.eval()
    hits: dict[str, list[float]] = defaultdict(list)
    for scene, q in items:
        out = generate(model, processor, image_fn(scene), q.short_prompt(), max_new_tokens=6)[0]
        hits[q.kind].append(exact_match(first_word(out), q.answer))
    acc = {k: sum(v) / len(v) for k, v in hits.items()}
    acc["mean"] = sum(acc.values()) / len(acc)
    return acc


def describe(scene: Scene) -> str:
    return "The image shows " + ", ".join(f"a {o.name}" for o in scene.objects) + "."


@torch.no_grad()
def text_count_accuracy(model, processor, items: list[tuple[Scene, Question]]) -> float:
    """Counting from a written list of objects, with a grey image.

    No perception is needed here: the objects are spelled out in the prompt.
    If RL lifts this score, it improved counting in language, not seeing.
    """
    model.eval()
    hits = []
    for scene, q in items:
        out = generate(model, processor, blank_like(scene), describe(scene) + " " + q.short_prompt(), max_new_tokens=6)[0]
        hits.append(exact_match(first_word(out), q.answer))
    return sum(hits) / len(hits)


@torch.no_grad()
def preference_accuracy(model, processor, pairs: list[PreferencePair], blind: bool = False,
                        batch_pairs: int = 8) -> dict[str, float]:
    """Fraction of pairs where log pi(chosen) > log pi(rejected).

    blind=True replaces every image with a grey square of the same size. A model
    that truly grounds its preference should fall to about 50% here.
    """
    model.eval()
    margins: dict[str, list[float]] = defaultdict(list)
    for s in range(0, len(pairs), batch_pairs):
        chunk = pairs[s: s + batch_pairs]
        images: list[Image.Image] | None = [blank_like(p.scene) for p in chunk] if blind else None
        c, r = pair_logprobs(model, pair_batch(processor, chunk, images).to(model.device))
        for p, m in zip(chunk, (c - r).tolist()):
            margins[p.kind].append(m)
    out = {k: sum(m > 0 for m in v) / len(v) for k, v in margins.items()}
    every = [m for v in margins.values() for m in v]
    out["mean"] = sum(m > 0 for m in every) / len(every)
    out["mean_margin"] = sum(every) / len(every)
    return out
