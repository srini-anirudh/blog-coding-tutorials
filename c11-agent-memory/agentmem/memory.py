"""AgentMemory: the write -> manage -> read loop in one object.

    observe(event)        WRITE   writer gate -> fact extraction -> ADD/UPDATE/SUPERSEDE/DELETE
    record_step(episode)  WRITE   raw (s, a, o, r, t) into the episodic log
    end_task(task)        MANAGE  summarise the trajectory into one retrievable episode
    end_session()         MANAGE  consolidate episodes into procedures, drop session notes
    retrieve(query)       READ    hybrid search over facts, episodes and procedures
    build_context(...)    READ    pack instructions, task state, memories, events and
                                  the latest observation under a token budget
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .budget import Block, ContextBudgeter, Packed
from .consolidate import consolidate, error_signature, find_recoveries
from .embed import Embedder, HashingEmbedder
from .episodic import EpisodicStore
from .procedural import Procedure, ProcedureStore
from .records import Episode, MemoryItem, skip_ids_in
from .retrieval import HybridRetriever, HybridWeights, Scored
from .semantic import Fact, FactStore, Op, RuleFactExtractor
from .working import TaskState
from .writer import MemoryEvent, MemoryWriter, WriteDecision


class AgentMemory:
    def __init__(self, embedder: Embedder | None = None, weights: HybridWeights | None = None,
                 use_writer: bool = True, supersede: bool = True, consolidate_min_support: int = 2,
                 scope_gate: bool = True) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.writer = MemoryWriter(self.embedder)
        self.extract = RuleFactExtractor()
        self.facts = FactStore()
        self.episodes = EpisodicStore()
        self.procedures = ProcedureStore()
        self.retriever = HybridRetriever(self.embedder, weights)
        self.budgeter = ContextBudgeter()
        self.working: TaskState | None = None
        self.session_notes: list[str] = []      # session-scoped statements: in context today, gone tomorrow
        self.use_writer = use_writer            # False: store everything (ablation)
        self.supersede = supersede              # False: append-only facts (ablation)
        self.consolidate_min_support = consolidate_min_support
        self.scope_gate = scope_gate            # False: "just for today" persists (ablation)
        self._item_of_fact: dict[str, str] = {}
        self._written: list[str] = []           # event texts the writer has accepted (novelty reference)

    # ----------------------------------------------------------------- write

    def observe(self, event: MemoryEvent) -> tuple[WriteDecision, list[tuple[Op | str, str]]]:
        """Run one event through the writer and, if it passes, into the stores."""
        claims = self.extract(event.text, event.t, event.source)
        edits = any(self.facts.decide(c) in (Op.SUPERSEDE, Op.DELETE) for c in claims)
        decision = self.writer.should_store(event, edits_memory=edits)
        if self.scope_gate and decision.scores.scope == "session":
            self.session_notes.append(event.text)
            return decision, []
        if self.use_writer and not decision.store:
            return decision, []
        ops: list[tuple[Op | str, str]] = []
        for claim in claims:
            if not self.supersede and claim.kind == "assert":
                fact = Fact(claim.subject, claim.attribute, claim.value or "", claim.t, claim.confidence,
                            valid_from=claim.t)
                self.facts.facts.append(fact)          # append-only: every version stays active
                ops.append((Op.ADD, self._index_fact(fact, decision.scores.importance)))
                continue
            op, fact = self.facts.apply(claim)
            ops.append((op, self._sync(op, fact, decision.scores.importance)))
        if not claims:  # no structured fact: keep the text itself as an episodic note
            item = MemoryItem("episodic", event.text, event.t, decision.scores.importance,
                              decision.scores.confidence, decision.scores.scope, event.source)
            self.retriever.add(item)
            ops.append(("NOTE", item.id))
        self.writer.remember(event.text)
        self._written.append(event.text)
        return decision, ops

    def _index_fact(self, fact: Fact, importance: float) -> str:
        item = MemoryItem("semantic", fact.text(), fact.t, importance, fact.confidence, "project",
                          fact.source, meta={"fact_id": fact.id})
        self.retriever.add(item)
        self._item_of_fact[fact.id] = item.id
        return item.id

    def _sync(self, op: Op, fact: Fact | None, importance: float) -> str:
        """Mirror a FactStore operation into the retrieval index."""
        if fact is None:
            return ""
        if op is Op.ADD:
            return self._index_fact(fact, importance)
        if op is Op.SUPERSEDE:
            old = next(f for f in self.facts.facts if f.superseded_by == fact.id)
            self.retriever.remove(self._item_of_fact.pop(old.id, ""))
            return self._index_fact(fact, importance)
        if op is Op.DELETE:
            item_id = self._item_of_fact.pop(fact.id, "")
            self.retriever.remove(item_id)
            return item_id
        if op is Op.UPDATE:
            item = self.retriever.items[self._item_of_fact[fact.id]]
            item.t, item.confidence = fact.t, fact.confidence
            return item.id
        return ""  # NOOP

    def record_step(self, episode: Episode) -> None:
        self.episodes.add(episode)

    # ----------------------------------------------------------------- manage

    def end_task(self, task: str) -> MemoryItem | None:
        """Summarise one finished trajectory into a single episodic memory."""
        steps = self.episodes.trajectory(task)
        if not steps:
            return None
        trace = "; ".join(f"`{e.action}` -> {error_signature(e.observation) or 'ok'}" for e in steps)
        reward = steps[-1].reward
        fixed = bool(find_recoveries(steps))
        item = MemoryItem("episodic", f"episode {task} (session {steps[0].session}): {trace}", steps[-1].t,
                          importance=0.9 if fixed else 0.4, source="agent",
                          meta={"task": task, "entity": steps[0].entity, "reward": reward})
        self.retriever.add(item)
        return item

    def end_session(self) -> list[Procedure]:
        """Consolidate, then forget what was only true for this session."""
        self.session_notes.clear()
        new = []
        for proc in consolidate(self.episodes, min_support=self.consolidate_min_support):
            known = self.procedures.lookup(proc.trigger, proc.error)
            merged = self.procedures.add(proc)
            item_id = f"proc-{merged.id}"
            self.retriever.remove(item_id)  # re-index with the updated support count
            self.retriever.add(MemoryItem("procedural", merged.text(), max(e.t for e in self.episodes),
                                          importance=1.0, source="consolidation", id=item_id))
            if known is None:
                new.append(merged)
        return new

    # ----------------------------------------------------------------- read

    def retrieve(self, query: str, now: float, k: int = 5) -> list[Scored]:
        return self.retriever.search(query, now, k)

    def build_context(self, instructions: str, request: str, now: float, budget: int,
                      recent_events: list[str] = (), observation: str | None = None, k: int = 8,
                      query: str | None = None) -> Packed:
        """Pack one context. `query` defaults to the request; pass the latest error too
        when the next decision depends on it."""
        blocks = [Block("instructions", instructions, required=True, compressible=False, id="instructions")]
        if self.working is not None:
            blocks.append(Block("task", self.working.render(), required=True, id="task"))
        for i, note in enumerate(self.session_notes):
            blocks.append(Block("task", f"today only: {note}", value=1.0, order=1 + i, id=f"note{i}"))
        blocks += self.memory_blocks(query or request, now, k)
        n = len(recent_events)
        for i, ev in enumerate(recent_events):
            # Newer events are worth more; the newest is worth as much as a strong memory.
            blocks.append(Block("events", ev, value=0.3 + 0.7 * (i + 1) / n, order=i, id=f"ev{i}"))
        if observation:
            blocks.append(Block("observation", observation, value=0.9, id="obs"))
        blocks.append(Block("request", request, required=True, compressible=False, id="request"))
        return self.budgeter.pack(blocks, budget)

    def memory_blocks(self, query: str, now: float, k: int = 8) -> list[Block]:
        """The retrieved memories as candidate blocks; the caller packs them."""
        return [Block("memory", f"- {hit.item.text}", value=hit.score, order=rank, id=hit.item.id)
                for rank, hit in enumerate(self.retrieve(query, now, k))]

    # ----------------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        state = {
            "facts": json.loads(self.facts.to_json()),
            "episodes": [asdict(e) for e in self.episodes],
            "procedures": [asdict(p) for p in self.procedures.procedures],
            "items": [asdict(it) for it in self.retriever.items.values()],
            "item_of_fact": self._item_of_fact,
            "written": self._written,
        }
        Path(path).write_text(json.dumps(state, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> "AgentMemory":
        state = json.loads(Path(path).read_text())
        mem = cls(embedder)
        skip_ids_in(x["id"] for key in ("episodes", "procedures", "items") for x in state[key])
        mem.facts = FactStore.from_json(json.dumps(state["facts"]))
        skip_ids_in(f.id for f in mem.facts.facts)
        for e in state["episodes"]:
            mem.episodes.add(Episode(**e))
        mem.procedures.procedures = [Procedure(**p) for p in state["procedures"]]
        for it in state["items"]:
            mem.retriever.add(MemoryItem(**it))
        for text in state["written"]:
            mem.writer.remember(text)
        mem._written = list(state["written"])
        mem._item_of_fact = state["item_of_fact"]
        return mem
