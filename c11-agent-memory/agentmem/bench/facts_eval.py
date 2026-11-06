"""Four ways to give the same reader the past, and a score for each.

    no_memory        only the current session's turns
    full_transcript  every turn so far, oldest cut first when it exceeds the budget
    retrieved        AgentMemory: written, managed and retrieved, packed by the budgeter
    oracle           exactly the gold fact, nothing else

The reader is held fixed across conditions, so any difference in accuracy is caused
by what the context contains. That is the causal claim: same policy, same tasks,
same seeds, one intervention.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from ..budget import Block, truncate_left
from ..memory import AgentMemory
from ..semantic import Claim, RuleFactExtractor
from ..tokens import count_tokens
from ..writer import MemoryEvent
from .facts_world import ATTRIBUTES, ENTITIES, Probe, Script, Turn

INSTRUCTIONS = ("You are the ops assistant for a software team. Answer questions about the team's projects "
                "from the context. Facts change over time: trust the latest confident statement, and ignore "
                "anything that was only true for an earlier day.")
CONDITIONS = ("no_memory", "full_transcript", "retrieved", "oracle")

_MEMORY_LINE = re.compile(r"^-?\s*(?P<subj>[a-z][\w-]*) (?P<attr>[a-z][a-z ]*?): `?(?P<val>[^`]+?)`?\s*$")
_TAG = re.compile(r"^\[s(?P<s>\d+)\]")


def render_turn(t: Turn) -> str:
    return f"[s{t.session}] {t.role}: {t.text}"


def request_for(p: Probe) -> str:
    return f"(session {p.session}) What is the {p.attribute} for {p.entity}?"


class RuleReader:
    """A careful, deterministic reader: it understands every phrasing in the world.

    It reads the context top to bottom. Confident statements and retractions set the
    answer; hedges count only if nothing confident was said; "just for today" counts
    only if it was said today. Given the right lines it is always right, so every
    error it makes is an error in what the context contained.
    """

    def __init__(self) -> None:
        self.extract = RuleFactExtractor()

    def __call__(self, context: str, entity: str, attribute: str, session: int) -> str:
        strong: str | None = None
        weak: str | None = None
        for line in context.splitlines():
            tag = _TAG.match(line)
            claims = self.extract(line, 0.0)
            m = _MEMORY_LINE.match(line)
            if not claims and m:
                claims = [_memory_claim(m)]
            for c in claims:
                if (c.subject, c.attribute) != (entity, attribute):
                    continue
                if c.scope == "session" and tag and int(tag["s"]) != session:
                    continue
                if c.kind == "delete":
                    strong, weak = "unknown", None
                elif c.confidence < 0.5:
                    weak = c.value
                else:
                    strong = c.value
        return strong or weak or "unknown"


def _memory_claim(m: re.Match[str]) -> Claim:
    return Claim(m["subj"], m["attr"].strip(), m["val"].strip(), 0.0)


@dataclass
class ProbeResult:
    probe: Probe
    condition: str
    answer: str
    correct: bool
    tokens: int


class CorruptedMemory(AgentMemory):
    """The causal check: same stores and retrieval, but every fact's value is wrong.

    If accuracy did not fall, the reader would not really be using its memory.
    """

    def _index_fact(self, fact, importance):
        pool = ATTRIBUTES.get(fact.attribute, ["wrong-value"])
        fact.value = next(v for v in pool + ["wrong-value"] if v != fact.value)
        return super()._index_fact(fact, importance)


def _entities(text: str) -> list[str]:
    return [e for e in ENTITIES if re.search(rf"\b{e}\b", text)]


class FactsEvaluator:
    """Builds the context for every probe of a script under every condition."""

    def __init__(self, budget: int, memory_factory: Callable[[], AgentMemory] = AgentMemory, k: int = 6) -> None:
        self.budget = budget
        self.memory_factory = memory_factory
        self.k = k

    def contexts(self, script: Script) -> Iterable[tuple[Probe, dict[str, str]]]:
        memory = self.memory_factory()
        fed, session = 0, 1
        for probe in script.probes:
            while fed < probe.after_turn:                      # stream turns into memory as they happen
                turn = script.turns[fed]
                if turn.session != session:
                    memory.end_session()
                    session = turn.session
                memory.observe(MemoryEvent(turn.text, "tool" if turn.role == "tool" else "user",
                                           turn.session + fed / 1000, entities=_entities(turn.text)))
                fed += 1
            seen = script.turns[: probe.after_turn]
            today = [render_turn(t) for t in seen if t.session == probe.session]
            yield probe, {
                "no_memory": self._transcript(today, probe),
                "full_transcript": self._transcript([render_turn(t) for t in seen], probe),
                "retrieved": memory.build_context(INSTRUCTIONS, request_for(probe), now=probe.session,
                                                  budget=self.budget, recent_events=today, k=self.k).render(),
                "oracle": self._oracle(probe),
            }

    def _transcript(self, lines: list[str], probe: Probe) -> str:
        fixed = count_tokens(INSTRUCTIONS) + count_tokens(request_for(probe)) + 12  # headers and newlines
        blocks = [Block("events", text, order=i) for i, text in enumerate(lines)]
        kept = truncate_left(blocks, self.budget - fixed)
        return (f"## Instructions\n{INSTRUCTIONS}\n\n## Conversation\n" + "\n".join(b.text for b in kept)
                + f"\n\n## Request\n{request_for(probe)}")

    def _oracle(self, probe: Probe) -> str:
        memory = f"\n\n## Relevant memory\n- {probe.entity} {probe.attribute}: {probe.gold}" \
            if probe.gold != "unknown" else ""
        return f"## Instructions\n{INSTRUCTIONS}{memory}\n\n## Request\n{request_for(probe)}"


def normalise(answer: str) -> str:
    return re.sub(r"[`'\".]", "", answer.strip().splitlines()[0] if answer.strip() else "").strip().lower()


def is_correct(answer: str, gold: str) -> bool:
    a = normalise(answer)
    if gold == "unknown":
        return a.startswith("unknown") or "not known" in a or "no record" in a
    return a == gold.lower() or a.startswith(gold.lower())


def run_facts(scripts: list[Script], budget: int, reader: Callable[[str, str, str, int], str] | None = None,
              memory_factory: Callable[[], AgentMemory] = AgentMemory,
              conditions: tuple[str, ...] = CONDITIONS) -> list[ProbeResult]:
    reader = reader or RuleReader()
    ev = FactsEvaluator(budget, memory_factory)
    results = []
    for script in scripts:
        for probe, ctx in ev.contexts(script):
            for cond in conditions:
                ans = reader(ctx[cond], probe.entity, probe.attribute, probe.session)
                results.append(ProbeResult(probe, cond, ans, is_correct(ans, probe.gold), count_tokens(ctx[cond])))
    return results


def summarise(results: list[ProbeResult]) -> dict[str, dict[str, float]]:
    """{condition: {"accuracy", "tokens", <kind>: accuracy}}."""
    by: dict[str, list[ProbeResult]] = defaultdict(list)
    for r in results:
        by[r.condition].append(r)
    out = {}
    for cond, rs in by.items():
        row = {"accuracy": sum(r.correct for r in rs) / len(rs), "tokens": sum(r.tokens for r in rs) / len(rs),
               "n": len(rs)}
        kinds: dict[str, list[bool]] = defaultdict(list)
        for r in rs:
            kinds[r.probe.kind].append(r.correct)
        row.update({k: sum(v) / len(v) for k, v in kinds.items()})
        out[cond] = row
    return out
