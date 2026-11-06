"""A synthetic world of tools, requests and calls, small enough to train a tiny model on.

Every tool has a made-up name (two syllable tokens), a purpose word, and one to three typed fields.
A request names a purpose and carries values. The right output is a call to the tool in context
whose purpose matches, with each value bound to the field of its type, or NONE if no tool matches.

Names are random, so a model that has learned to *read* the tool list can call a tool it never saw.
A model that has *memorised* the training tools cannot. That is the post's lookup-versus-interpretation
question, made small enough to measure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

CONSONANTS = "bdfgklmnprstvz"
VOWELS = "aeiou"
SYLLABLES = [c + v for c in CONSONANTS for v in VOWELS]  # 70 syllables -> 4,900 two-syllable names

PURPOSES = [
    "weather", "book", "convert", "translate", "search", "play", "order", "cancel",
    "remind", "phone", "pay", "track", "rate", "share", "list", "delete",
]
VALUES = {
    "city": ["paris", "malmo", "tokyo", "lima", "oslo", "cairo", "delhi", "rome",
             "quito", "perth", "accra", "hanoi", "dakar", "riga", "sofia", "porto"],
    "day": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow"],
    "count": [f"#{i}" for i in range(1, 21)],
    "person": ["asha", "bram", "chen", "dina", "emeka", "farah", "goran", "hana",
               "ivo", "jun", "kemal", "lena", "mira", "nils", "omar", "priya"],
    "item": ["pizza", "ticket", "lamp", "novel", "song", "taxi", "table", "coffee",
             "jacket", "phone_case", "umbrella", "guitar", "bike", "camera", "kettle", "chair"],
    "lang": ["english", "hindi", "swedish", "tamil", "french", "swahili", "korean", "spanish", "arabic", "polish"],
}
TYPES = list(VALUES)
FILLER = ["please", "for", "quickly", "now", "can", "you"]

SPECIALS = ["<pad>", "<bos>", "<eos>", "<tools>", "<tool>", "<user>", "<call>", "NONE",
            ":", "(", ")", ",", "="]


@dataclass(frozen=True)
class Tool:
    name: tuple[str, str]
    purpose: str
    fields: tuple[tuple[tuple[str, str], str], ...]  # ((field_name, type), ...) in schema order

    @property
    def name_str(self) -> str:
        return "".join(self.name)

    def field_types(self) -> dict[str, str]:
        return {"".join(f): t for f, t in self.fields}


@dataclass
class Example:
    tools: list[Tool]  # what the application put in the context
    request: list[str]  # the user's words
    gold: Tool | None  # the tool to call, or None for "no tool applies"
    args: dict[str, str] = field(default_factory=dict)  # field name -> value

    def prompt_tokens(self) -> list[str]:
        out = ["<bos>", "<tools>"]
        for t in self.tools:
            out += render_tool(t)
        return out + ["<user>", *self.request, "<call>"]

    def target_tokens(self) -> list[str]:
        if self.gold is None:
            return ["NONE", "<eos>"]
        return [*render_call(self.gold, self.args), "<eos>"]


def render_tool(t: Tool) -> list[str]:
    out = ["<tool>", *t.name, ":", t.purpose, "("]
    for k, (fname, ftype) in enumerate(t.fields):
        if k:
            out.append(",")
        out += [*fname, ":", ftype]
    return out + [")"]


def render_call(t: Tool, args: dict[str, str]) -> list[str]:
    out = [*t.name, "("]
    for k, (fname, _) in enumerate(t.fields):
        if k:
            out.append(",")
        out += [*fname, "=", args["".join(fname)]]
    return out + [")"]


class Vocab:
    def __init__(self) -> None:
        words = SPECIALS + SYLLABLES + PURPOSES + TYPES + FILLER + [v for t in TYPES for v in VALUES[t]]
        assert len(words) == len(set(words)), "token collision in the synthetic vocabulary"
        self.itos = words
        self.stoi = {w: i for i, w in enumerate(words)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, toks: list[str]) -> list[int]:
        return [self.stoi[t] for t in toks]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]


def split_names(seed: int = 0, n_heldout: int = 600) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """All two-syllable names, split into a training pool and a held-out pool that never appears in training."""
    names = [(a, b) for a in SYLLABLES for b in SYLLABLES]
    random.Random(seed).shuffle(names)
    return names[n_heldout:], names[:n_heldout]


def make_tools(n: int, names: list[tuple[str, str]], rng: random.Random) -> list[Tool]:
    """n tools whose names and field names are drawn from `names` (without reuse)."""
    pool = list(names)
    rng.shuffle(pool)
    tools, k = [], 0
    for i in range(n):
        n_fields = rng.randint(1, 3)
        types = rng.sample(TYPES, n_fields)
        tool_name = pool[k % len(pool)]
        fnames = [pool[(k + 1 + j) % len(pool)] for j in range(n_fields)]
        k += 1 + n_fields
        tools.append(Tool(tool_name, PURPOSES[i % len(PURPOSES)] if i < len(PURPOSES) else rng.choice(PURPOSES),
                          tuple(zip(fnames, types))))
    return tools


def make_example(pool: list[Tool], rng: random.Random, max_tools: int = 3, p_none: float = 0.2,
                 n_tools: int | None = None) -> Example:
    """Put 1..max_tools tools with distinct purposes in context and write one request."""
    k = n_tools or rng.randint(1, max_tools)
    ctx: list[Tool] = []
    seen: set[str] = set()
    for t in rng.sample(pool, min(len(pool), 8 * k)):
        if t.purpose not in seen:
            ctx.append(t)
            seen.add(t.purpose)
        if len(ctx) == k:
            break
    if rng.random() < p_none:
        # No tool applies: the request asks for a purpose that nothing in context serves.
        purpose = rng.choice([p for p in PURPOSES if p not in seen])
        types = rng.sample(TYPES, rng.randint(1, 3))
        words = [purpose] + [rng.choice(VALUES[t]) for t in types]
        gold, args = None, {}
    else:
        gold = rng.choice(ctx)
        args = {"".join(f): rng.choice(VALUES[t]) for f, t in gold.fields}
        words = [gold.purpose] + list(args.values())
    head, rest = words[0], words[1:]
    rng.shuffle(rest)
    words = [head] + rest
    for _ in range(rng.randint(0, 2)):
        words.insert(rng.randint(0, len(words)), rng.choice(FILLER))
    return Example(ctx, words, gold, args)


# ---- Parse, Exec, e: the three functions that live in the harness, not the model ----

def parse_call(toks: list[str], tools: list[Tool] | None = None) -> dict | None:
    """Turn an emitted span into {'name': str, 'args': {field: value}}, or {'none': True}. None if malformed."""
    if toks[-1:] == ["<eos>"]:
        toks = toks[:-1]
    if toks == ["NONE"]:
        return {"none": True}
    if len(toks) < 4 or toks[2] != "(" or toks[-1] != ")":
        return None
    if not (toks[0] in SYLLABLES and toks[1] in SYLLABLES):
        return None
    name, body = toks[0] + toks[1], toks[3:-1]
    args: dict[str, str] = {}
    parts, cur = [], []
    for t in body:
        if t == ",":
            parts.append(cur)
            cur = []
        else:
            cur.append(t)
    parts.append(cur)
    for p in parts:
        if len(p) != 4 or p[2] != "=" or not (p[0] in SYLLABLES and p[1] in SYLLABLES):
            return None
        args[p[0] + p[1]] = p[3]
    return {"name": name, "args": args}


def value_type(v: str) -> str | None:
    for t in TYPES:
        if v in VALUES[t]:
            return t
    return None


def execute(call: dict, tools: list[Tool]) -> str:
    """A fake backend. It runs a call against the tools the application exposed, or raises."""
    by_name = {t.name_str: t for t in tools}
    if call["name"] not in by_name:
        raise KeyError(f"no such tool: {call['name']}")
    tool = by_name[call["name"]]
    types = tool.field_types()
    if set(call["args"]) != set(types):
        raise ValueError(f"fields {sorted(call['args'])} do not match schema {sorted(types)}")
    for f, v in call["args"].items():
        if value_type(v) != types[f]:
            raise TypeError(f"field {f} wants a {types[f]}, got {v!r}")
    payload = ",".join(f"{f}={v}" for f, v in sorted(call["args"].items()))
    return f"{tool.purpose}-ok[{payload}]"


def encode_result(result: str) -> str:
    """e: serialise the tool's result back into text the model will read on the next pass."""
    return f"<result> {result} </result>"
