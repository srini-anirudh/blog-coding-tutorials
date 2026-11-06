"""A toy visual world where every fact is known, and the language prior is built in on purpose.

A scene is a set of objects. Each present object has a colour and a count. Scenes come from three
templates (street, park, house), so some objects co-occur: a dog is common in a street scene, and
cars are usually white. Those regularities are the language prior a model can learn from text alone.

The "image" is not pixels. It is a (K, F) tensor, one row per object slot:
    [present, colour one-hot (4), count one-hot (3)]
scaled by an evidence strength and buried in Gaussian noise. That keeps the model tiny and makes
image-side negatives exact: a crop removes one slot, an edit changes one fact.

A response is a token sequence of claims, "two red car ;" per object, optionally with style words
("lovely", "shiny", ...) in front. Style carries no information about the image.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

OBJECTS = ("car", "person", "dog", "door", "tree", "sign", "bike", "cat")
COLOURS = ("white", "red", "black", "blue")
COUNTS = ("one", "two", "three")
STYLE = ("lovely", "shiny", "charming", "vivid")
SPECIAL = ("<pad>", "<bos>", "<eos>", ";", "<describe>")
VOCAB = SPECIAL + OBJECTS + COLOURS + COUNTS + STYLE
TOK = {w: i for i, w in enumerate(VOCAB)}
PAD, BOS, EOS, SEP, DESCRIBE = (TOK[w] for w in SPECIAL)

K = len(OBJECTS)                       # object slots in an image
F = 1 + len(COLOURS) + len(COUNTS)     # features per slot

# Templates give co-occurrence: P(object present | template).
TEMPLATES = {
    "street": {"car": 0.9, "person": 0.8, "door": 0.6, "sign": 0.5, "dog": 0.5},
    "park": {"tree": 0.9, "person": 0.7, "bike": 0.6, "dog": 0.6, "cat": 0.2},
    "house": {"door": 0.9, "cat": 0.6, "tree": 0.5, "car": 0.4, "person": 0.3},
}
# Colour priors per object. Cars are mostly white; the decoder will learn that.
COLOUR_PRIOR = {
    "car": (0.55, 0.15, 0.2, 0.1), "person": (0.25, 0.25, 0.25, 0.25), "dog": (0.5, 0.1, 0.35, 0.05),
    "door": (0.6, 0.2, 0.1, 0.1), "tree": (0.1, 0.3, 0.5, 0.1), "sign": (0.4, 0.4, 0.1, 0.1),
    "bike": (0.2, 0.3, 0.3, 0.2), "cat": (0.3, 0.1, 0.5, 0.1),
}
COUNT_PRIOR = (0.6, 0.3, 0.1)


@dataclass(frozen=True)
class Claim:
    obj: str
    colour: str
    count: str


@dataclass
class Scene:
    facts: dict[str, Claim]   # present objects only
    template: str

    def claims(self) -> list[Claim]:
        return list(self.facts.values())


def sample_scene(rng: random.Random) -> Scene:
    template = rng.choice(list(TEMPLATES))
    present = [o for o, p in TEMPLATES[template].items() if rng.random() < p]
    if not present:
        present = [max(TEMPLATES[template], key=TEMPLATES[template].get)]
    facts = {}
    for o in present:
        colour = rng.choices(COLOURS, COLOUR_PRIOR[o])[0]
        facts[o] = Claim(o, colour, rng.choices(COUNTS, COUNT_PRIOR)[0])
    return Scene(facts, template)


def clean_image(scene: Scene) -> torch.Tensor:
    """(K, F) noise-free evidence."""
    x = torch.zeros(K, F)
    for c in scene.claims():
        k = OBJECTS.index(c.obj)
        x[k, 0] = 1.0
        x[k, 1 + COLOURS.index(c.colour)] = 1.0
        x[k, 1 + len(COLOURS) + COUNTS.index(c.count)] = 1.0
    return x


def render(scene: Scene, gen: torch.Generator, noise: float = 0.8, strength: float = 1.0) -> torch.Tensor:
    """What the model sees: strength * clean + noise * N(0, 1)."""
    return strength * clean_image(scene) + noise * torch.randn(K, F, generator=gen)


def null_image() -> torch.Tensor:
    """No visual evidence at all (the 'grey image'). The model sees it during SFT, so P(y | x, null) is its prior."""
    return torch.zeros(K, F)


def crop(img: torch.Tensor, slots: list[int], gen: torch.Generator, noise: float = 0.8) -> torch.Tensor:
    """Remove the evidence in some slots, as if that part of the picture were cut away. The noise stays."""
    out = img.clone()
    for k in slots:
        out[k] = noise * torch.randn(F, generator=gen)
    return out


def edit_colour(scene: Scene, obj: str, colour: str) -> Scene:
    """The same scene with one fact changed: the V-DPO-style targeted image edit."""
    facts = dict(scene.facts)
    c = facts[obj]
    facts[obj] = Claim(obj, colour, c.count)
    return Scene(facts, scene.template)


def encode(claims: list[Claim], style: list[str | None] | None = None) -> list[int]:
    """<bos> <describe> [style] count colour object ; ... <eos>"""
    ids = [BOS, DESCRIBE]
    for i, c in enumerate(claims):
        if style and style[i]:
            ids.append(TOK[style[i]])
        ids += [TOK[c.count], TOK[c.colour], TOK[c.obj], SEP]
    return ids + [EOS]


PROMPT_LEN = 2  # <bos> <describe>: never scored


def decode(ids: list[int]) -> tuple[list[Claim], int]:
    """Parse generated ids into claims. Returns (claims, number of style words). Malformed claims are dropped."""
    words = [VOCAB[i] for i in ids]
    if words[:PROMPT_LEN] == ["<bos>", "<describe>"]:
        words = words[PROMPT_LEN:]
    claims, n_style, cur = [], 0, []
    for w in words:
        if w == "<eos>":
            break
        if w in STYLE:
            n_style += 1
        elif w == ";":
            if len(cur) == 3 and cur[0] in COUNTS and cur[1] in COLOURS and cur[2] in OBJECTS:
                claims.append(Claim(cur[2], cur[1], cur[0]))
            cur = []
        else:
            cur.append(w)
    return claims, n_style


def style_for(n: int, rng: random.Random, p: float) -> list[str | None]:
    return [rng.choice(STYLE) if rng.random() < p else None for _ in range(n)]


def hallucinate(scene: Scene, claims: list[Claim], kind: str, rng: random.Random) -> list[Claim]:
    """Introduce exactly one error of `kind` into a correct claim list.

    object: insert a claim about an absent object, chosen by co-occurrence with the template (a hard negative).
    object_swap: replace one present object with an absent one, so the length does not change.
    colour: swap the colour of one present object. count: change one count.
    """
    claims = list(claims)
    absent = [o for o in TEMPLATES[scene.template] if o not in scene.facts] or \
             [o for o in OBJECTS if o not in scene.facts]
    if kind == "object":
        o = rng.choice(absent)
        new = Claim(o, rng.choices(COLOURS, COLOUR_PRIOR[o])[0], rng.choices(COUNTS, COUNT_PRIOR)[0])
        claims.insert(rng.randrange(len(claims) + 1), new)
        return claims
    i = rng.randrange(len(claims))
    c = claims[i]
    if kind == "object_swap":  # same length: one present object replaced by an absent one
        claims[i] = Claim(rng.choice(absent), c.colour, c.count)
    elif kind == "colour":
        claims[i] = Claim(c.obj, rng.choice([x for x in COLOURS if x != c.colour]), c.count)
    elif kind == "count":
        claims[i] = Claim(c.obj, c.colour, rng.choice([x for x in COUNTS if x != c.count]))
    else:
        raise ValueError(kind)
    return claims
