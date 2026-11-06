"""Section 18, "How do you know it works?": the five-condition ablation ladder.

A small world of made-up projects, each deployed to a region, with some later moves. The
history is a list of dated notes. Every question asks where one project deploys now. The
projects are invented, so a model cannot know the answers from its weights.

Five conditions, read by the same model with the same prompt:
  1 no_memory   the question only
  2 random      k notes about other projects, same format and count as retrieved
  3 retrieved   the "actual system": bag-of-words cosine top-k, no recency, no supersession,
                shown in time order
  4 oracle      the one note that answers the question
  5 full        every note, in order

C11 runs four of these (not random) through its own fact world. This module adds the random
condition, the gap decomposition and the "did it already know" control.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass

PROJECTS = ["kestrel", "marlin", "osprey", "quokka", "tamarin", "vireo", "wombat", "zander", "bittern", "caracal",
            "dunnock", "egret", "fossa", "gannet", "hoopoe", "ibex", "jacana", "kinkajou", "lemming", "margay",
            "numbat", "ocelot", "pika", "quetzal"]
REGIONS = ["eu-west-2", "ap-south-1", "us-east-1", "sa-east-1", "ca-central-1", "af-south-1", "me-south-1", "ap-northeast-3"]
NOISE = ["the user prefers short replies", "the staging runners were upgraded", "the on-call rotation changed",
         "the user asked for a weekly summary", "a flaky test was quarantined", "the team renamed the main branch"]
CONDITIONS = ["no_memory", "random", "retrieved", "oracle", "full"]
SYSTEM = "You answer questions about the user's projects. Use the memory notes when they are given."


@dataclass(frozen=True)
class Note:
    day: int
    text: str
    project: str | None = None


@dataclass(frozen=True)
class Question:
    project: str
    gold: str
    moved: bool


def make_world(seed: int, n_projects: int = 24, moved_share: float = 0.5) -> tuple[list[Note], list[Question]]:
    rng = random.Random(seed)
    projects = rng.sample(PROJECTS, n_projects)
    notes: list[Note] = []
    first = {p: rng.choice(REGIONS) for p in projects}
    for p in projects:
        notes.append(Note(rng.randint(1, 40), f"project {p} deploys to {first[p]}.", p))
    questions = []
    movers = set(rng.sample(projects, int(n_projects * moved_share)))
    for p in projects:
        gold = first[p]
        if p in movers:
            gold = rng.choice([r for r in REGIONS if r != first[p]])
            day = rng.randint(45, 90)
            notes.append(Note(day, f"project {p} moved and now deploys to {gold}.", p))
        questions.append(Question(p, gold, p in movers))
    for _ in range(n_projects // 2):
        notes.append(Note(rng.randint(1, 90), rng.choice(NOISE) + "."))
    notes.sort(key=lambda n: n.day)
    return notes, questions


def _bag(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9-]+", text.lower()))


def bow_cosine(a: str, b: str) -> float:
    x, y = _bag(a), _bag(b)
    dot = sum(x[t] * y[t] for t in x)
    return dot / math.sqrt(sum(v * v for v in x.values()) * sum(v * v for v in y.values()) or 1)


def question_text(q: Question) -> str:
    return f"Which region does project {q.project} deploy to now?"


def fmt(notes: list[Note]) -> str:
    return "\n".join(f"- day {n.day}: {n.text}" for n in notes)


def memory_for(cond: str, q: Question, notes: list[Note], k: int, rng: random.Random) -> list[Note]:
    if cond == "no_memory":
        return []
    if cond == "full":
        return list(notes)
    mine = [n for n in notes if n.project == q.project]
    if cond == "oracle":
        return [max(mine, key=lambda n: n.day)]
    if cond == "random":
        others = [n for n in notes if n.project not in (None, q.project)]
        return sorted(rng.sample(others, k), key=lambda n: n.day)
    if cond == "retrieved":
        qt = question_text(q)
        ranked = sorted(notes, key=lambda n: (-bow_cosine(qt, n.text), n.day))  # no recency term
        return sorted(ranked[:k], key=lambda n: n.day)  # shown in time order, like every other condition
    raise ValueError(cond)


def prompt_for(q: Question, memory: list[Note]) -> str:
    ask = f"Question: {question_text(q)}\nReply with the region code only."
    if not memory:
        return ask
    return f"Memory notes:\n{fmt(memory)}\n\n{ask}"


def decompose(acc: dict[str, float]) -> dict[str, float]:
    """The post's four gaps, from accuracies A1..A5 of the five conditions."""
    a1, a2, a3, a4, a5 = (acc[c] for c in CONDITIONS)
    return {
        "distraction cost A2-A1": a2 - a1,
        "delivered lift A3-A1": a3 - a1,
        "retrieval headroom A4-A3": a4 - a3,
        "value of right memories A4-A1": a4 - a1,
        "compression loss A5-A4": a5 - a4,
    }
