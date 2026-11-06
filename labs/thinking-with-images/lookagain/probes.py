"""Diagnostic probes and the sampling trap, written against any reader with QwenVLM's interface."""
from __future__ import annotations

import random

from .answers import CAPTION, QUESTION, VERIFY, closing_time, majority, times_in, weighted_majority
from .scene import CLOSE_TIMES, GRID, PRIOR_CLOSE, Scene
from .views import View, cell_view, full_view, random_view, render


def batched(vlm, items, bs=16, **kw) -> list[list[str]]:
    out = []
    for i in range(0, len(items), bs):
        out += vlm.generate(items[i:i + bs], **kw)
    return out


def blind_prior(vlm) -> str | None:
    """What the model answers with no image at all. The question is the same for every photo, so one call is enough."""
    return closing_time(vlm.ask([], QUESTION))


def score(scenes: list[Scene], preds: list[str | None], prior: str | None = PRIOR_CLOSE) -> dict:
    """Accuracy, plus where the wrong answers came from: the language prior, the bus sign, or no time at all.

    `prior` is the blind answer from `blind_prior`; "said_prior" is the share of answers equal to it.
    """
    n = len(scenes)
    return {
        "acc": sum(p == s.close_time for s, p in zip(scenes, preds)) / n,
        "said_prior": sum(p == prior for p in preds) / n,
        "said_bus_time": sum(p == s.bus_time for s, p in zip(scenes, preds)) / n,
        "no_time": sum(p is None for p in preds) / n,
        "n": n,
    }


def ask_views(vlm, scenes, views: list[View] | View | None, prompt=QUESTION, **kw) -> list[str | None]:
    """One greedy answer per scene. `views=None` is the blind probe: no image at all."""
    if views is None:
        items = [([], prompt) for _ in scenes]
    else:
        vs = views if isinstance(views, list) else [views] * len(scenes)
        items = [([render(s, v)], prompt) for s, v in zip(scenes, vs)]
    return [closing_time(a[0]) for a in batched(vlm, items, **kw)]


def oracle_swap(vlm, scenes, **kw) -> list[str | None]:
    """Replace the image with a text statement of the facts (not the answer)."""
    items = [([], s.facts + " " + QUESTION) for s in scenes]
    return [closing_time(a[0]) for a in batched(vlm, items, **kw)]


def caption_then_reason(vlm, scenes, view: View, **kw) -> tuple[list[str | None], list[bool]]:
    """The model transcribes the text it sees; then the same model answers from the transcript alone.

    Also returns whether the true closing time appears anywhere in the transcript.
    """
    caps = [a[0] for a in batched(vlm, [([render(s, view)], CAPTION) for s in scenes], max_new_tokens=64, **kw)]
    items = [([], f"Text read from a street photo: \"{c}\". {QUESTION}") for c in caps]
    preds = [closing_time(a[0]) for a in batched(vlm, items, **kw)]
    has = [s.close_time in times_in(c) for s, c in zip(scenes, caps)]
    return preds, has


def answer_region_view(scene: Scene, width=448) -> View:
    return cell_view(*scene.plaque_cell, width=width)


def samples_vs_views(vlm, scenes, k=16, width=448, temperature=1.0, seed=0, bs=4):
    """k text samples from one low-resolution look, against k greedy answers from k different random views.

    Returns per-scene lists of parsed answers for both, so pass@k and majority@k can be computed for any k' <= k.
    """
    look = full_view(width)
    samples = []
    for i in range(0, len(scenes), bs):
        chunk = scenes[i:i + bs]
        out = vlm.generate([([render(s, look)], QUESTION) for s in chunk], sample=True,
                           temperature=temperature, n=k, seed=seed + i)
        samples += [[closing_time(a) for a in o] for o in out]
    rng = random.Random(seed)
    views, weights = [], []
    for s in scenes:
        vs = [random_view(rng, width=width) for _ in range(k)]
        imgs = [render(s, v) for v in vs]
        out = batched(vlm, [([im], QUESTION) for im in imgs])
        views.append([closing_time(o[0]) for o in out])
        weights.append([w for i in range(0, k, 16) for w in vlm.yes_prob([([im], VERIFY) for im in imgs[i:i + 16]])])
    return samples, views, weights


def pass_and_vote(scenes, answers, ks, weights=None) -> dict:
    """pass@k: any of the first k answers is right. vote@k: the majority of the first k is right.

    With `weights` (one list per scene), also wvote@k: the vote where each answer counts for its weight.
    """
    out = {}
    for k in ks:
        out[k] = {
            "pass": sum(s.close_time in a[:k] for s, a in zip(scenes, answers)) / len(scenes),
            "vote": sum(majority(a[:k]) == s.close_time for s, a in zip(scenes, answers)) / len(scenes),
        }
        if weights is not None:
            out[k]["wvote"] = sum(weighted_majority(a[:k], w[:k]) == s.close_time
                                  for s, a, w in zip(scenes, answers, weights)) / len(scenes)
    return out


def chance_pass(k: int, n_options: int = len(CLOSE_TIMES)) -> float:
    """pass@k of a reader that guesses uniformly among the possible closing times, independently each time."""
    return 1 - (1 - 1 / n_options) ** k


def all_cells(width=448):
    return [cell_view(c, r, GRID, width) for r in range(GRID) for c in range(GRID)]


def bus_cell_view(scene: Scene, width=448) -> View:
    """The 4 x 4 grid cell that holds the bus sign."""
    return next(v for v in (cell_view(c, r, GRID, width) for r in range(GRID) for c in range(GRID))
                if v.contains(scene.bus_box))


def verifier_separation(vlm, scenes, prompt) -> dict:
    """P(Yes) of a verification prompt on the plaque cell and on the bus-sign cell, over all scenes."""
    on_plaque = vlm.yes_prob([([render(s, answer_region_view(s))], prompt) for s in scenes])
    on_bus = vlm.yes_prob([([render(s, bus_cell_view(s))], prompt) for s in scenes])
    n = len(scenes)
    return {"p_yes_plaque": sum(on_plaque) / n, "p_yes_bus": sum(on_bus) / n,
            "plaque_above_bus": sum(a > b for a, b in zip(on_plaque, on_bus)) / n,
            "bus_above_half": sum(b > 0.5 for b in on_bus) / n, "plaque_below_half": sum(a <= 0.5 for a in on_plaque) / n}
