"""Hybrid retrieval: rank memories by meaning, age and importance together.

    S = alpha * S_semantic + beta * S_recency + gamma * S_importance

    S_semantic   = max(0, cos(query, memory))            in [0, 1]
    S_recency    = 0.5 ** ((now - t_memory) / half_life)  in (0, 1]
    S_importance = importance written by the writer       in [0, 1]

This is the scoring rule of Generative Agents (Park et al., 2023). They min-max
normalise each term over the candidates; we keep fixed ranges instead, so a score
means the same thing from one query to the next and the weights stay interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .embed import Embedder
from .index_fast import TensorIndex
from .records import Kind, MemoryItem


@dataclass
class HybridWeights:
    alpha: float = 1.0      # semantic
    beta: float = 0.5       # recency
    gamma: float = 0.3      # importance
    half_life: float = 3.0  # in logical time units (sessions)


@dataclass
class Scored:
    item: MemoryItem
    score: float
    semantic: float
    recency: float
    importance: float


class HybridRetriever:
    def __init__(self, embedder: Embedder, weights: HybridWeights | None = None) -> None:
        self.embedder = embedder
        self.weights = weights or HybridWeights()
        self.index = TensorIndex(embedder.dim)
        self.items: dict[str, MemoryItem] = {}

    def add(self, item: MemoryItem) -> None:
        self.items[item.id] = item
        self.index.add([item.id], self.embedder.encode([item.text]))

    def remove(self, item_id: str) -> None:
        """Superseded and deleted memories leave the index; they stay in the store's history."""
        if item_id in self.items:
            del self.items[item_id]
            self.index.remove(item_id)

    def components(self, query: str, now: float) -> tuple[list[MemoryItem], torch.Tensor]:
        """Every live memory and its three score terms, [N, 3] = (semantic, recency, importance)."""
        q = self.embedder.encode([query])                                          # [1, D]
        sims = self.index.scores(q)[0]                                             # [size], dead = -inf
        slots = [self.index.slot_of[i] for i in self.items]
        items = list(self.items.values())
        semantic = sims[slots].clamp(min=0.0)                                      # [N]
        ages = torch.tensor([now - it.t for it in items], dtype=torch.float32)
        recency = 0.5 ** (ages.clamp(min=0) / self.weights.half_life)              # [N]
        importance = torch.tensor([it.importance for it in items], dtype=torch.float32)
        return items, torch.stack([semantic, recency, importance], dim=-1)

    def search(self, query: str, now: float, k: int = 5, kinds: set[Kind] | None = None,
               mmr_lambda: float | None = None) -> list[Scored]:
        if not self.items:
            return []
        items, comp = self.components(query, now)
        w = torch.tensor([self.weights.alpha, self.weights.beta, self.weights.gamma], dtype=torch.float32)
        total = comp @ w                                                            # [N]
        keep = [i for i, it in enumerate(items) if it.status == "active" and (kinds is None or it.kind in kinds)]
        order = sorted(keep, key=lambda i: -float(total[i]))
        if mmr_lambda is not None:
            order = self._mmr(items, order[: 4 * k], total, k, mmr_lambda)
        return [Scored(items[i], float(total[i]), *map(float, comp[i])) for i in order[:k]]

    def _mmr(self, items: list[MemoryItem], order: list[int], total: torch.Tensor, k: int,
             lam: float) -> list[int]:
        """Maximal marginal relevance: trade score against similarity to what is already picked."""
        vecs = self.embedder.encode([items[i].text for i in order])                 # [C, D]
        picked: list[int] = []
        while order and len(picked) < k:
            def mmr(pos: int) -> float:
                redundancy = max((float(vecs[pos] @ vecs[p]) for p in picked), default=0.0)
                return lam * float(total[order[pos]]) - (1 - lam) * redundancy
            best = max((p for p in range(len(order)) if p not in picked), key=mmr, default=None)
            if best is None:
                break
            picked.append(best)
        return [order[p] for p in picked]
