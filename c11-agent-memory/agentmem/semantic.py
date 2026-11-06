"""Semantic memory: stable facts, keyed by (subject, attribute), with an edit history.

Facts arrive as `Claim`s from an extractor. `FactStore.apply` turns each claim into
exactly one operation:

    ADD        no fact for this key yet
    UPDATE     same value again: refresh the time, raise confidence
    SUPERSEDE  a new value replaces the old one; the old one is closed, not erased
    DELETE     the user retracted the fact; a tombstone records when
    NOOP       a weak claim contradicts a strong fact; keep the fact, log the conflict

The point is the last four. An append-only store keeps both "port 8080" and
"port 9090" forever and leaves the model to guess which is current.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Literal

from .records import new_id


class Op(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    SUPERSEDE = "SUPERSEDE"
    DELETE = "DELETE"
    NOOP = "NOOP"


@dataclass
class Claim:
    """One extracted statement, before the store has judged it."""

    subject: str
    attribute: str
    value: str | None              # None for a retraction
    t: float
    confidence: float = 0.9
    scope: str = "project"
    kind: Literal["assert", "delete"] = "assert"
    source: str = "user"
    evidence: str = ""


@dataclass
class Fact:
    subject: str
    attribute: str
    value: str
    t: float
    confidence: float = 0.9
    source: str = "user"
    status: Literal["active", "superseded", "deleted"] = "active"
    valid_to: float | None = None
    superseded_by: str | None = None
    support: int = 1
    valid_from: float | None = None   # when it became true; `t` is when it was last confirmed
    id: str = field(default_factory=lambda: new_id("f"))

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject, self.attribute)

    def text(self) -> str:
        return f"{self.subject} {self.attribute}: {self.value}"


def _same(a: str, b: str) -> bool:
    return a.strip("` ").casefold() == b.strip("` ").casefold()


class FactStore:
    """Facts plus the log of every operation applied to them."""

    def __init__(self, override_margin: float = 0.3) -> None:
        # A contradicting claim must be at most this much less confident than the
        # current fact to replace it. 0.3 lets a confident user statement (0.9)
        # overwrite another (0.9) but stops a hedge (0.4) from overwriting either.
        self.override_margin = override_margin
        self.facts: list[Fact] = []
        self.log: list[tuple[Op, Claim, str | None]] = []   # (op, claim, id of the fact touched)

    def current(self, subject: str, attribute: str) -> Fact | None:
        for f in reversed(self.facts):
            if f.key == (subject, attribute) and f.status == "active":
                return f
        return None

    def active(self) -> list[Fact]:
        return [f for f in self.facts if f.status == "active"]

    def history(self, subject: str, attribute: str) -> list[Fact]:
        return sorted((f for f in self.facts if f.key == (subject, attribute)), key=lambda f: f.valid_from)

    def as_of(self, subject: str, attribute: str, t: float) -> Fact | None:
        """What we believed at time t: the fact whose validity interval contains t."""
        for f in self.history(subject, attribute):
            if f.valid_from <= t and (f.valid_to is None or t < f.valid_to):
                return f
        return None

    def decide(self, claim: Claim) -> Op:
        """Which operation `claim` would cause, without applying it."""
        cur = self.current(claim.subject, claim.attribute)
        if claim.kind == "delete":
            return Op.NOOP if cur is None else Op.DELETE
        if cur is None:
            return Op.ADD
        if _same(cur.value, claim.value or ""):
            return Op.UPDATE
        if claim.confidence + self.override_margin < cur.confidence:
            return Op.NOOP  # a weak contradiction: keep the fact, log the claim
        return Op.SUPERSEDE

    def apply(self, claim: Claim) -> tuple[Op, Fact | None]:
        op = self.decide(claim)
        cur = self.current(claim.subject, claim.attribute)
        if op is Op.ADD:
            return self._record(op, claim, self._insert(claim))
        if op is Op.UPDATE:
            cur.t = claim.t
            cur.confidence = 1 - (1 - cur.confidence) * (1 - claim.confidence)  # noisy-OR corroboration
            cur.support += 1
        elif op is Op.SUPERSEDE:
            new = self._insert(claim)
            cur.status, cur.valid_to, cur.superseded_by = "superseded", claim.t, new.id
            return self._record(op, claim, new)
        elif op is Op.DELETE:
            cur.status, cur.valid_to = "deleted", claim.t
        return self._record(op, claim, cur)

    def conflicts(self) -> list[Claim]:
        """Claims that contradicted a fact and lost. Worth surfacing to a human."""
        return [c for op, c, fid in self.log if op is Op.NOOP and fid is not None]

    def _insert(self, claim: Claim) -> Fact:
        fact = Fact(claim.subject, claim.attribute, claim.value or "", claim.t, claim.confidence, claim.source,
                    valid_from=claim.t)
        self.facts.append(fact)
        return fact

    def _record(self, op: Op, claim: Claim, fact: Fact | None) -> tuple[Op, Fact | None]:
        self.log.append((op, claim, fact.id if fact else None))
        return op, fact

    def to_json(self) -> str:
        return json.dumps([asdict(f) for f in self.facts], sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "FactStore":
        store = cls()
        store.facts = [Fact(**d) for d in json.loads(s)]
        return store


# --- extraction --------------------------------------------------------------------

_VALUE = r"(?P<val>`[^`]+`|[\w./:@-]+)"
_ATTR = r"(?P<attr>[a-z][a-z ]*?)"
_SUBJ = r"(?P<subj>[a-z][\w-]*)"
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\bforget the {_ATTR} (?:for|of) {_SUBJ}\b", re.I), "delete"),
    (re.compile(rf"\b(?:moved|changed|switched) the {_ATTR} (?:for|of) {_SUBJ} to {_VALUE}", re.I), "assert"),
    (re.compile(rf"\bthe {_ATTR} (?:for|of) {_SUBJ} (?:is now|is|might be) {_VALUE}", re.I), "assert"),
    (re.compile(rf"\b{_SUBJ}'s {_ATTR} is {_VALUE}", re.I), "assert"),
]
HEDGE_RE = re.compile(r"\b(i think|maybe|might|probably|not sure|i guess)\b", re.I)
TRANSIENT_RE = re.compile(r"\b(just for today|for today|for now|this time|temporarily|right now)\b", re.I)


class RuleFactExtractor:
    """Regex extraction for the phrasings the benchmark world uses.

    Real systems ask an LLM to do this step (see `LLMFactExtractor`). Rules keep the
    benchmark exact: every extraction error is visible and reproducible.
    """

    def __call__(self, text: str, t: float, source: str = "user") -> list[Claim]:
        claims = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            for pattern, kind in _PATTERNS:
                m = pattern.search(sentence)
                if not m:
                    continue
                hedged = bool(HEDGE_RE.search(sentence))
                claims.append(Claim(
                    subject=m["subj"].lower(), attribute=m["attr"].lower().strip(),
                    value=None if kind == "delete" else m["val"].rstrip(".,").strip("`"),
                    t=t, confidence=0.4 if hedged else 0.9,
                    scope="session" if TRANSIENT_RE.search(sentence) else "project",
                    kind=kind, source=source, evidence=sentence))
                break  # one claim per sentence; patterns are ordered most specific first
        return claims


EXTRACT_PROMPT = """Extract durable facts from the message as a JSON list.
Each item: {{"subject": str, "attribute": str, "value": str or null, "op": "assert" or "delete", "confidence": 0-1}}.
Use "delete" when the user retracts a fact. Lower confidence for hedged statements. Return [] if none.

Message: {text}
JSON:"""


class LLMFactExtractor:
    """Ask a model for claims as JSON, then validate every field before trusting it."""

    def __init__(self, complete: Callable[[str], str]) -> None:
        self.complete = complete

    def __call__(self, text: str, t: float, source: str = "user") -> list[Claim]:
        raw = self.complete(EXTRACT_PROMPT.format(text=text))
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return []  # a malformed extraction must cost us a memory, never corrupt the store
        claims = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict) or not isinstance(it.get("subject"), str) or not isinstance(it.get("attribute"), str):
                continue
            kind = "delete" if it.get("op") == "delete" else "assert"
            value = it.get("value")
            if kind == "assert" and not isinstance(value, str):
                continue
            conf = it.get("confidence", 0.7)
            conf = min(1.0, max(0.0, float(conf))) if isinstance(conf, (int, float)) else 0.7
            claims.append(Claim(it["subject"].lower(), it["attribute"].lower(), value if kind == "assert" else None,
                                t, conf, "project", kind, source, text))
        return claims
