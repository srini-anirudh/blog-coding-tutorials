"""Section 10, "Update is not append", and section 17's "Deletion has to be real".

Three stores answer the same questions:

- AppendStore keeps every assertion and returns all that match.
- ReplaceStore keeps one value per (subject, relation) and overwrites it.
- BitemporalStore keeps two clocks per row: when the fact was true in the world
  (valid_from, valid_to) and when the system held the row (recorded_at, retracted_at).
  Rows are never edited. A change retracts the old row and records a new version, so the
  store can answer both "what was true then" and "what did we believe then".
  It has SUPERSEDE ("the world changed"), INVALIDATE ("we were wrong") and DELETE. Derived
  records (summaries) list their sources, so a delete can chase them.

C11's SemanticMemory already does ADD/UPDATE/SUPERSEDE/DELETE on one clock. This module adds
the second clock, INVALIDATE, and derivation-aware deletion.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

FOREVER = date.max


@dataclass(frozen=True)
class Row:
    fact: int  # lineage id: every version of one assertion shares it
    subject: str
    relation: str
    value: str
    valid_from: date
    valid_to: date  # exclusive; FOREVER while the fact is believed still true
    recorded_at: date
    retracted_at: date = FOREVER
    source: str = "user"

    def believed(self, known_at: date) -> bool:
        return self.recorded_at <= known_at and (known_at < self.retracted_at or self.retracted_at == FOREVER)


@dataclass(frozen=True)
class Derived:
    """A record built from other records, such as a rolling profile summary."""

    text: str
    sources: tuple[int, ...]  # fact ids


class AppendStore:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def tell(self, subject, relation, value, **_) -> None:
        self.rows.append((subject, relation, value))

    def current(self, subject, relation) -> list[str]:
        return [v for s, r, v in self.rows if (s, r) == (subject, relation)]


class ReplaceStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], str] = {}

    def tell(self, subject, relation, value, **_) -> None:
        self.rows[(subject, relation)] = value

    def current(self, subject, relation) -> list[str]:
        v = self.rows.get((subject, relation))
        return [] if v is None else [v]


@dataclass
class BitemporalStore:
    rows: list[Row] = field(default_factory=list)
    derived: list[Derived] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    _facts: int = 0

    def _new_version(self, old: Row, at: date, **changes) -> Row:
        """Retract `old` from `at` on and record a changed copy at `at`."""
        self.rows[self.rows.index(old)] = replace(old, retracted_at=at)
        new = replace(old, recorded_at=at, retracted_at=FOREVER, **changes)
        self.rows.append(new)
        return new

    def _open(self, subject, relation, known_at: date) -> list[Row]:
        return [r for r in self.rows if (r.subject, r.relation) == (subject, relation)
                and r.believed(known_at) and r.valid_to == FOREVER]

    def tell(self, subject, relation, value, valid_from: date, recorded_at: date, source: str = "user") -> int:
        """ADD a fact, or SUPERSEDE the open fact for the same slot when the value differs."""
        for old in self._open(subject, relation, recorded_at):
            if old.value == value:
                self.log.append(f"NOOP       {subject}.{relation}={value} already known")
                return old.fact
            self._new_version(old, recorded_at, valid_to=valid_from)
            self.log.append(f"SUPERSEDE  {subject}.{relation}={old.value} closed at {valid_from} (learned {recorded_at})")
        self._facts += 1
        self.rows.append(Row(self._facts, subject, relation, value, valid_from, FOREVER, recorded_at, source=source))
        self.log.append(f"ADD        {subject}.{relation}={value} valid from {valid_from} (learned {recorded_at}, {source})")
        return self._facts

    def invalidate(self, fact: int, at: date) -> None:
        """INVALIDATE: the fact was never true. Its rows are retracted, not erased, and the
        value it wrongly closed is reopened."""
        bad = [r for r in self.rows if r.fact == fact and r.believed(at)]
        for r in bad:
            self.rows[self.rows.index(r)] = replace(r, retracted_at=at)
        b = bad[0]
        self.log.append(f"INVALIDATE {b.subject}.{b.relation}={b.value} from source '{b.source}' (at {at})")
        for r in list(self.rows):
            if (r.subject, r.relation) == (b.subject, b.relation) and r.believed(at) and r.valid_to == b.valid_from:
                self._new_version(r, at, valid_to=FOREVER)

    def current(self, subject, relation, today: date = FOREVER) -> list[str]:
        return [r.value for r in self._open(subject, relation, today)]

    def as_of(self, subject, relation, world_time: date, known_at: date = FOREVER) -> list[str]:
        """The value true in the world on `world_time`, by what the store held on `known_at`.

        known_at=FOREVER asks "what do we now think was true then". An earlier known_at asks
        "what did we believe at the time", which is what explains a past decision.
        """
        return [r.value for r in self.rows if (r.subject, r.relation) == (subject, relation)
                and r.believed(known_at) and r.valid_from <= world_time < r.valid_to]

    def derive(self, text: str, facts: list[int]) -> None:
        self.derived.append(Derived(text, tuple(facts)))

    def delete(self, fact: int, chase_derived: bool = True) -> int:
        """DELETE for real: drop every version of the fact and, if asked, every record derived from it."""
        self.rows = [r for r in self.rows if r.fact != fact]
        before = len(self.derived)
        if chase_derived:
            self.derived = [d for d in self.derived if fact not in d.sources]
        removed = before - len(self.derived)
        self.log.append(f"DELETE     fact #{fact}; derived records removed: {removed}")
        return removed

    def visible_text(self, today: date = FOREVER) -> str:
        """Everything a later context could quote: believed rows plus derived summaries."""
        facts = [f"{r.subject} {r.relation} {r.value}" for r in self.rows if r.believed(today)]
        return "\n".join(facts + [d.text for d in self.derived])
