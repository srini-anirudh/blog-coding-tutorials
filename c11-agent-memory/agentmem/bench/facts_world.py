"""A reproducible multi-session world where the right answer depends on the past.

A team talks to its ops assistant over several sessions. Facts about projects are
stated, changed, hedged about, overridden for one day, and retracted. Between them
sit chit-chat and long tool logs, as in a real transcript. At the end of each
session we ask probe questions whose answers were given in earlier sessions.

Every probe has an exact gold answer, computed by replaying the ground truth. The
probe kinds isolate one memory skill each:

    stated       said once, never changed                    -> recall
    updated      changed later ("we moved ... to ...")       -> supersede
    rumor        a hedge contradicts a confident statement   -> contradiction handling
    deleted      retracted ("forget the ...")                -> delete; gold is "unknown"
    today        overridden "just for today", asked same day -> session scope
    after_today  the day after an override                   -> session facts must not persist
    unseen       never mentioned                             -> abstain; gold is "unknown"
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

ENTITIES = ["atlas", "borealis", "cobalt", "dune", "ember", "fjord", "granite", "harbor", "iris",
            "juniper", "kestrel", "lumen"]
ATTRIBUTES: dict[str, list[str]] = {
    "deploy region": ["eu-west-1", "us-east-2", "ap-south-1", "eu-central-1", "us-west-2", "sa-east-1"],
    "test command": ["make test", "pytest -x", "tox -e py312", "npm test", "cargo test", "go test"],
    "owner": ["priya", "marco", "chen", "amara", "jonas", "leila"],
    "database": ["postgres", "mysql", "sqlite", "mongodb", "redis", "cassandra"],
    "release branch": ["main", "release", "stable", "prod", "trunk", "live"],
}
CHATTER = [
    "Thanks, that helps.", "Can you remind me what we discussed about the roadmap?",
    "Great, let's keep going.", "Sounds good to me.", "I will be out tomorrow afternoon.",
    "The design review moved to Thursday.", "Ok, next item.", "Let's grab lunch after this.",
]


@dataclass
class Turn:
    session: int
    role: str          # "user" or "tool"
    text: str


@dataclass
class Probe:
    session: int
    entity: str
    attribute: str
    gold: str          # a value, or "unknown"
    kind: str
    after_turn: int    # index into the flat turn list: the probe sees turns[:after_turn]
    age: int = 0       # sessions since the key was last mentioned


@dataclass
class Script:
    turns: list[Turn] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)


def _fmt(value: str) -> str:
    return f"`{value}`" if " " in value else value


def _log(rng: random.Random, session: int) -> str:
    """A noisy tool log: the kind of text that fills a transcript without adding facts."""
    svc = rng.choice(["api", "worker", "scheduler", "gateway"])
    lines = [f"$ kubectl logs {svc}-{rng.randint(1000, 9999)} --tail 8"]
    for _ in range(8):
        ms = rng.randint(3, 900)
        lines.append(f"{rng.randint(10, 23)}:{rng.randint(10, 59)}:{rng.randint(10, 59)} INFO {svc} "
                     f"request id={rng.getrandbits(40):010x} status=200 latency={ms}ms")
    return "\n".join(lines)


def make_script(seed: int, n_sessions: int = 8, n_entities: int = 6, facts_per_early_session: int = 4,
                logs_per_session: int = 2, chatter_per_session: int = 3) -> Script:
    rng = random.Random(seed)
    entities = rng.sample(ENTITIES, n_entities)
    keys = [(e, a) for e in entities for a in ATTRIBUTES]
    rng.shuffle(keys)
    truth: dict[tuple[str, str], str] = {}          # durable value of each key, or absent
    history: dict[tuple[str, str], list[str]] = {}  # event kinds that touched each key
    last_seen: dict[tuple[str, str], int] = {}      # session of the latest mention
    script = Script()
    unseen = [k for k in keys[: n_sessions]]       # a few keys that are never mentioned
    pool = keys[n_sessions:]

    def say(s: int, text: str, role: str = "user") -> None:
        script.turns.append(Turn(s, role, text))

    def pick_value(key: tuple[str, str], exclude: str | None = None) -> str:
        return rng.choice([v for v in ATTRIBUTES[key[1]] if v != exclude])

    for s in range(1, n_sessions + 1):
        events: list[tuple[str, tuple[str, str] | None]] = []
        if s <= 3:  # early sessions mostly introduce facts
            events += [("state", pool.pop()) for _ in range(facts_per_early_session) if pool]
        else:
            known = [k for k in truth]
            rng.shuffle(known)
            for kind in ("update", "rumor", "today", "delete", "state"):
                if kind == "state" and pool:
                    events.append((kind, pool.pop()))
                elif known and kind != "state":
                    events.append((kind, known.pop()))
        events += [("chatter", None)] * chatter_per_session + [("log", None)] * logs_per_session
        rng.shuffle(events)
        today: dict[tuple[str, str], str] = {}
        for kind, key in events:
            if key is not None:
                last_seen[key] = s
            if kind == "chatter":
                say(s, rng.choice(CHATTER))
            elif kind == "log":
                say(s, _log(rng, s), role="tool")
            elif kind == "state":
                assert key is not None
                truth[key] = pick_value(key)
                template = rng.choice(["The {a} for {e} is {v}.", "{e}'s {a} is {v}."])
                say(s, template.format(a=key[1], e=key[0], v=_fmt(truth[key])))
                history.setdefault(key, []).append("state")
            elif kind == "update":
                truth[key] = pick_value(key, truth[key])
                say(s, f"Heads up: we moved the {key[1]} for {key[0]} to {_fmt(truth[key])}.")
                history[key].append("update")
            elif kind == "rumor":
                say(s, f"I think the {key[1]} for {key[0]} might be {_fmt(pick_value(key, truth[key]))}, not sure.")
                history[key].append("rumor")
            elif kind == "today":
                today[key] = pick_value(key, truth[key])
                say(s, f"Just for today, the {key[1]} for {key[0]} is {_fmt(today[key])}.")
                history[key].append("today")
            elif kind == "delete":
                del truth[key]
                say(s, f"Forget the {key[1]} for {key[0]}, we no longer use it.")
                history[key].append("delete")
        # Probes at the end of the session, about anything said so far, plus an unseen key.
        n = len(script.turns)
        if s >= 2:
            touched = [k for k in history if k not in today]
            rng.shuffle(touched)
            for key in touched[:4]:
                last = history[key][-1]
                kind = {"state": "stated", "update": "updated", "rumor": "rumor", "delete": "deleted",
                        "today": "after_today"}[last]
                script.probes.append(Probe(s, *key, truth.get(key, "unknown"), kind, n, s - last_seen[key]))
            for key, value in today.items():
                script.probes.append(Probe(s, *key, value, "today", n))
            if unseen:
                key = unseen.pop()
                script.probes.append(Probe(s, *key, "unknown", "unseen", n))
    return script
