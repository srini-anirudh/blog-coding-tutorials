"""Multimodal preference records (I, x, y_w, y_l) in two flavours.

style   y_w is plain and correct; y_l has one error AND style words. Like pairs where a strong model wrote
        the winner and a weaker one the loser: text alone separates them, so a learner never has to look.
fact    y_w and y_l have the same style and length and differ in exactly one fact (an object swap, a colour
        or a count). Only the image can tell them apart. Used to test what training learned, and to train a
        reward model that must look.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .world import STYLE, Scene, encode, hallucinate, render, sample_scene, style_for

KINDS = ("object", "colour", "count")


@dataclass
class Pairs:
    scenes: list[Scene]
    imgs: torch.Tensor       # (N, K, F)
    yw: list[list[int]]
    yl: list[list[int]]
    kinds: list[str]

    def __len__(self) -> int:
        return len(self.scenes)


def build_pairs(n: int, seed: int, flavour: str, noise: float = 0.8, kinds=KINDS) -> Pairs:
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    out = Pairs([], [], [], [], [])
    for _ in range(n):
        s = sample_scene(rng)
        good = s.claims()
        rng.shuffle(good)
        kind = rng.choice(kinds)
        bad = hallucinate(s, good, kind, rng)
        if flavour == "style":
            w = encode(good)
            st = style_for(len(bad), rng, 0.5)
            if not any(st):
                st[rng.randrange(len(st))] = rng.choice(STYLE)
            lo = encode(bad, st)
        elif flavour == "fact":
            if kind == "object":
                bad = hallucinate(s, good, "object_swap", rng)
            st = style_for(len(good), rng, 0.3)
            w, lo = encode(good, st), encode(bad, st)
        else:
            raise ValueError(flavour)
        out.scenes.append(s)
        out.imgs.append(render(s, gen, noise))
        out.yw.append(w)
        out.yl.append(lo)
        out.kinds.append(kind)
    out.imgs = torch.stack(out.imgs)
    return out
