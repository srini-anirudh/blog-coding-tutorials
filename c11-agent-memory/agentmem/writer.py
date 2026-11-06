"""The memory writer: decide whether an event deserves to become a long-term memory.

`should_store(event)` scores five things and combines them:

    novelty     1 - max cosine to what is already stored      (do we know this?)
    stability   will it still be true next session?           (a port vs. a PID)
    importance  will getting it wrong cost something?         (explicit asks, errors, reward)
    confidence  how much do we trust the source?              (user > tool > model guess)
    scope       global / project / session                    (session facts never persist)

Scope, novelty and confidence are gates. Stability, importance and confidence then
form a weighted score against a threshold. The weights below are hand-set and
documented, not learned; the tests pin the decisions on a fixed set of events so a
change in weights is visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch

from .embed import Embedder
from .semantic import HEDGE_RE, TRANSIENT_RE

Source = Literal["user", "tool", "model"]

_STABLE_RE = re.compile(r"\b(always|never|prefer|default|is|uses|policy|convention|owner|command)\b", re.I)
_EXPLICIT_RE = re.compile(r"\b(remember|important|must|do not|don't|never|always|note that)\b", re.I)
_ERROR_RE = re.compile(r"\b(\w+Error|\w+Exception|failed|traceback)\b", re.I)
_VOLATILE_RE = re.compile(r"\b(pid \d+|\d{1,2}:\d{2}(:\d{2})?|[0-9a-f]{12,}|\d+(\.\d+)? ?(ms|s)\b)", re.I)
_CHATTER_RE = re.compile(r"^\s*(thanks|thank you|ok|okay|great|sounds good|hi|hello)\b", re.I)


@dataclass
class MemoryEvent:
    """Anything the agent saw: a user turn, a tool result, or its own conclusion."""

    text: str
    source: Source
    t: float
    reward: float | None = None
    entities: list[str] = field(default_factory=list)


@dataclass
class WriteScores:
    novelty: float
    stability: float
    importance: float
    confidence: float
    scope: str


@dataclass
class WriteDecision:
    store: bool
    score: float
    scores: WriteScores
    reason: str


class MemoryWriter:
    def __init__(self, embedder: Embedder, weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
                 threshold: float = 0.5, min_novelty: float = 0.1, min_confidence: float = 0.5,
                 importance_fn: Callable[[MemoryEvent], float] | None = None) -> None:
        self.embedder = embedder
        self.weights = weights  # (stability, importance, confidence)
        self.threshold = threshold
        self.min_novelty = min_novelty
        self.min_confidence = min_confidence
        self.importance_fn = importance_fn  # e.g. an LLM rating, as in Generative Agents
        self.stored: torch.Tensor = torch.zeros(0, embedder.dim)  # [M, D] vectors of stored memories

    def remember(self, text: str) -> None:
        """Tell the writer a memory now exists, so the same text is not novel again."""
        self.stored = torch.cat([self.stored, self.embedder.encode([text])])

    def score(self, event: MemoryEvent) -> WriteScores:
        text = event.text
        v = self.embedder.encode([text])[0]                                  # [D]
        novelty = 1.0 if len(self.stored) == 0 else 1.0 - float((self.stored @ v).max().clamp(min=0))

        base_stability = {"user": 0.6, "tool": 0.35, "model": 0.3}[event.source]
        stability = base_stability + 0.25 * bool(_STABLE_RE.search(text)) \
            - 0.4 * bool(TRANSIENT_RE.search(text)) - 0.25 * bool(_VOLATILE_RE.search(text)) \
            - 0.4 * bool(_CHATTER_RE.search(text))

        if self.importance_fn is not None:
            importance = self.importance_fn(event)
        else:
            importance = 0.3 + 0.3 * bool(_EXPLICIT_RE.search(text)) + 0.3 * bool(_ERROR_RE.search(text))
            if event.reward is not None:
                importance += 0.3 * abs(event.reward)

        confidence = {"user": 0.9, "tool": 0.85, "model": 0.5}[event.source]
        if HEDGE_RE.search(text):
            confidence *= 0.5

        if TRANSIENT_RE.search(text):
            scope = "session"
        elif event.entities:
            scope = "project"
        else:
            scope = "global"
        clip = lambda x: min(1.0, max(0.0, x))  # noqa: E731
        return WriteScores(clip(novelty), clip(stability), clip(importance), clip(confidence), scope)

    def should_store(self, event: MemoryEvent, edits_memory: bool = False) -> WriteDecision:
        """`edits_memory`: the event would change or retract a stored fact. Such an event
        is never "not novel", even when its wording repeats an earlier one."""
        s = self.score(event)
        ws, wi, wc = self.weights
        score = ws * s.stability + wi * s.importance + wc * s.confidence
        if s.scope == "session":
            return WriteDecision(False, score, s, "session-scoped: keep in working memory only")
        if s.novelty < self.min_novelty and not edits_memory:
            return WriteDecision(False, score, s, f"not novel (novelty {s.novelty:.2f})")
        if s.confidence < self.min_confidence:
            return WriteDecision(False, score, s, f"too uncertain (confidence {s.confidence:.2f})")
        if score < self.threshold:
            return WriteDecision(False, score, s, f"score {score:.2f} < threshold {self.threshold}")
        return WriteDecision(True, score, s, f"score {score:.2f} >= threshold {self.threshold}")
