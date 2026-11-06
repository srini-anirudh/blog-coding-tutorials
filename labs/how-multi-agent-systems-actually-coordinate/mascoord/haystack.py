"""Synthetic record stores and three ways to read them: one agent, or a coordinator with workers.

Two tasks, built so the post's two regimes can both be measured:

- lookup (breadth-first). One record holds the answer: "Project amber-falcon: access code 4821."
  Every other record is a distractor of the same shape. Splitting the store across workers
  loses nothing, because the answer lives in one record.
- two-hop (interdependent). "Project amber-falcon is led by Mira Okafor." and, somewhere else,
  "Mira Okafor works in room 312." The question asks for the room of the project's leader.
  A worker that sees one fact cannot know it needs the other.

Protocols take an `ask(prompts) -> replies` function, so tests can run them with a rule-based
stub and the scripts can run them with a real model.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

ADJ = ["amber", "azure", "brisk", "cedar", "coral", "dusky", "ember", "frost", "golden", "hazel",
       "ivory", "jade", "lunar", "maple", "misty", "noble", "olive", "pale", "quiet", "rapid",
       "ruby", "sable", "silver", "solar", "stone", "swift", "tawny", "umber", "velvet", "wild"]
NOUN = ["falcon", "otter", "heron", "lynx", "badger", "raven", "marten", "ibis", "viper", "bison",
        "crane", "gecko", "hare", "kestrel", "lemur", "moth", "newt", "orca", "puma", "quail",
        "robin", "shrike", "tapir", "urchin", "vole", "wren", "yak", "zebu", "egret", "finch"]
FIRST = ["Mira", "Tomas", "Ayla", "Rohan", "Lena", "Kofi", "Ines", "Yuki", "Omar", "Sana",
         "Bram", "Nadia", "Pavel", "Zara", "Iker", "Leila", "Dmitri", "Asha", "Felix", "Noor"]
LAST = ["Okafor", "Lindqvist", "Moreau", "Haddad", "Petrov", "Nakamura", "Alvarez", "Brennan",
        "Castillo", "Dubois", "Eriksen", "Fischer", "Gupta", "Ivanova", "Jensen", "Kowalski",
        "Laine", "Mendes", "Novak", "Ortega", "Pillai", "Quinn", "Rossi", "Sato",
        "Tanaka", "Ueda", "Varga", "Weber", "Xu", "Yilmaz", "Zielinski", "Abara"]

Ask = Callable[[list[str]], list[str]]


@dataclass
class Instance:
    task: str                  # "lookup" or "two-hop"
    records: list[str]
    question: str
    answer: str
    target: str                # the project name asked about
    bridge: str = ""           # two-hop only: the leader's name
    meta: dict = field(default_factory=dict)


def _unique(rng: random.Random, n: int, make: Callable[[], str]) -> list[str]:
    seen: list[str] = []
    for _ in range(200 * n):
        if len(seen) == n:
            break
        x = make()
        if x not in seen:
            seen.append(x)
    if len(seen) < n:
        raise ValueError(f"name pool too small for {n} unique values")
    return seen


def make_lookup(n_records: int, rng: random.Random) -> Instance:
    names = _unique(rng, n_records, lambda: f"{rng.choice(ADJ)}-{rng.choice(NOUN)}")
    codes = _unique(rng, n_records, lambda: str(rng.randint(1000, 9999)))
    records = [f"Project {p}: access code {c}." for p, c in zip(names, codes)]
    i = rng.randrange(n_records)
    q = f"What is the access code for project {names[i]}?"
    return Instance("lookup", records, q, codes[i], names[i], meta={"position": i / max(1, n_records - 1)})


def make_two_hop(n_records: int, rng: random.Random) -> Instance:
    """n_records // 2 lead facts and n_records // 2 room facts, shuffled together."""
    half = n_records // 2
    names = _unique(rng, half, lambda: f"{rng.choice(ADJ)}-{rng.choice(NOUN)}")
    people = _unique(rng, half, lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}")
    rooms = _unique(rng, half, lambda: str(rng.randint(100, 999)))
    lead = [f"Project {p} is led by {who}." for p, who in zip(names, people)]
    order = list(range(half))
    rng.shuffle(order)
    room = [f"{people[j]} works in room {rooms[j]}." for j in order]
    records = lead + room
    rng.shuffle(records)
    i = rng.randrange(half)
    q = f"Which room does the leader of project {names[i]} work in?"
    return Instance("two-hop", records, q, rooms[i], names[i], bridge=people[i])


# --- prompts ----------------------------------------------------------------------------

def context_prompt(records: list[str], question: str, instruction: str) -> str:
    body = "\n".join(records)
    return f"Records:\n{body}\n\nQuestion: {question}\n{instruction}"


ANSWER_ONLY = "Reply with only the number. If the records do not contain the answer, reply NONE."
WHO_LEADS = "Reply with only the person's full name. If the records do not say, reply NONE."


def parse_number(reply: str) -> str | None:
    if "NONE" in reply.upper()[:8]:
        return None
    m = re.search(r"\b(\d{3,4})\b", reply)
    return m.group(1) if m else None


def parse_name(reply: str, known: list[str]) -> str | None:
    """The first known full name that appears in the reply."""
    for name in known:
        if name in reply:
            return name
    return None


def shards(records: list[str], k: int) -> list[list[str]]:
    """Contiguous, near-equal slices. Every record goes to exactly one worker."""
    n = len(records)
    bounds = [round(i * n / k) for i in range(k + 1)]
    return [records[bounds[i]:bounds[i + 1]] for i in range(k)]


def aggregate(values: list[str | None]) -> tuple[str | None, int]:
    """Coordinator rule: the most common non-NONE value. Returns (value, n_distinct_values)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, 0
    best = max(set(vals), key=lambda v: (vals.count(v), -vals.index(v)))
    return best, len(set(vals))


def checked(values: list[str | None], slices: list[list[str]], key: str) -> list[str | None]:
    """A non-model verifier: keep a worker's value only if its own slice has a record that
    names `key` and contains the value. This is "evidence pointer, verify before propagate"."""
    out = []
    for v, sl in zip(values, slices):
        ok = v is not None and any(key in r and v in r for r in sl)
        out.append(v if ok else None)
    return out


# --- protocols --------------------------------------------------------------------------

def solo(inst: Instance, ask: Ask) -> dict:
    reply = ask([context_prompt(inst.records, inst.question, ANSWER_ONLY)])[0]
    got = parse_number(reply)
    return {"answer": got, "correct": got == inst.answer, "rounds": 1}


def split_one_round(inst: Instance, ask: Ask, k: int, verify: bool = False) -> dict:
    """Coordinator fans the original question out to k workers, one round, then aggregates."""
    sl = shards(inst.records, k)
    replies = ask([context_prompt(s, inst.question, ANSWER_ONLY) for s in sl])
    vals = [parse_number(r) for r in replies]
    if verify:
        key = inst.target if inst.task == "lookup" else ""
        vals = checked(vals, sl, key) if key else vals
    got, distinct = aggregate(vals)
    return {"answer": got, "correct": got == inst.answer, "rounds": 1, "conflict": distinct > 1}


def split_two_round(inst: Instance, ask: Ask, k: int, people: list[str]) -> dict:
    """The coordinator plans the dependency: first find the leader, then that person's room."""
    sl = shards(inst.records, k)
    q1 = f"Who leads project {inst.target}?"
    names = [parse_name(r, people) for r in ask([context_prompt(s, q1, WHO_LEADS) for s in sl])]
    who, _ = aggregate(names)
    if who is None:
        return {"answer": None, "correct": False, "rounds": 1, "bridge_ok": False}
    q2 = f"Which room does {who} work in?"
    rooms = [parse_number(r) for r in ask([context_prompt(s, q2, ANSWER_ONLY) for s in sl])]
    got, distinct = aggregate(rooms)
    return {"answer": got, "correct": got == inst.answer, "rounds": 2,
            "bridge_ok": who == inst.bridge, "conflict": distinct > 1}


def all_people() -> list[str]:
    return [f"{f} {l}" for f in FIRST for l in LAST]


# --- a rule-based reader, for tests and for a perfect-reader reference ------------------

def oracle_ask(prompts: list[str]) -> list[str]:
    """Reads the prompt the way a perfect agent would. Two-hop questions need both facts."""
    out = []
    for p in prompts:
        recs = p.split("Records:\n", 1)[1].split("\n\nQuestion: ", 1)[0].split("\n")
        q = p.split("\n\nQuestion: ", 1)[1].split("\n", 1)[0]
        ans = "NONE"
        if m := re.match(r"What is the access code for project (\S+)\?", q):
            ans = next((r.split()[-1].rstrip(".") for r in recs if f"Project {m.group(1)}:" in r), "NONE")
        elif m := re.match(r"Who leads project (\S+)\?", q):
            ans = next((r.split(" is led by ")[1].rstrip(".") for r in recs if f"Project {m.group(1)} is" in r), "NONE")
        elif m := re.match(r"Which room does the leader of project (\S+) work in\?", q):
            who = next((r.split(" is led by ")[1].rstrip(".") for r in recs if f"Project {m.group(1)} is" in r), None)
            if who:
                ans = next((r.split()[-1].rstrip(".") for r in recs if r.startswith(who + " works")), "NONE")
        elif m := re.match(r"Which room does (.+) work in\?", q):
            ans = next((r.split()[-1].rstrip(".") for r in recs if r.startswith(m.group(1) + " works")), "NONE")
        out.append(ans)
    return out
