"""Exact vector index as one matrix: search is a matmul plus top-k.

Rows live in a preallocated [capacity, D] tensor that doubles when full, so adding
one memory is amortised O(D) instead of a full copy. Removal flips a mask bit;
removed rows score -inf and the slot is reused by the next add.

This is still exact (brute-force) search. At a few hundred thousand memories a
graph index such as HNSW starts to pay for itself; below that, one matmul wins.
"""

from __future__ import annotations

import torch


class TensorIndex:
    def __init__(self, dim: int, capacity: int = 1024, device: str = "cpu") -> None:
        self.dim = dim
        self.device = device
        self.matrix = torch.zeros(capacity, dim, device=device)   # [C, D]
        self.alive = torch.zeros(capacity, dtype=torch.bool, device=device)  # [C]
        self.slot_of: dict[str, int] = {}
        self.id_of: list[str | None] = [None] * capacity
        self.free: list[int] = []
        self.size = 0  # high-water mark of used slots

    def _grow(self, need: int) -> None:
        cap = self.matrix.shape[0]
        if need <= cap:
            return
        new_cap = max(need, 2 * cap)
        self.matrix = torch.cat([self.matrix, torch.zeros(new_cap - cap, self.dim, device=self.device)])
        self.alive = torch.cat([self.alive, torch.zeros(new_cap - cap, dtype=torch.bool, device=self.device)])
        self.id_of += [None] * (new_cap - cap)

    def add(self, ids: list[str], vectors: torch.Tensor) -> None:
        """vectors: [N, D], already L2-normalised. Re-adding an id overwrites it."""
        for id_, v in zip(ids, vectors.to(self.device)):
            if id_ in self.slot_of:
                slot = self.slot_of[id_]
            elif self.free:
                slot = self.free.pop()
            else:
                self._grow(self.size + 1)
                slot = self.size
                self.size += 1
            self.matrix[slot] = v
            self.alive[slot] = True
            self.slot_of[id_] = slot
            self.id_of[slot] = id_

    def remove(self, id_: str) -> None:
        slot = self.slot_of.pop(id_)
        self.alive[slot] = False
        self.id_of[slot] = None
        self.free.append(slot)

    def scores(self, queries: torch.Tensor) -> torch.Tensor:
        """queries: [Q, D] -> cosine for every slot, [Q, size]; dead slots are -inf."""
        sims = queries.to(self.device) @ self.matrix[: self.size].T          # [Q, size]
        return sims.masked_fill(~self.alive[: self.size], float("-inf"))

    def search(self, query: torch.Tensor, k: int) -> list[tuple[str, float]]:
        """query: [D]. Same contract as NaiveIndex.search."""
        return self.search_batch(query.unsqueeze(0), k)[0]

    def search_batch(self, queries: torch.Tensor, k: int) -> list[list[tuple[str, float]]]:
        """queries: [Q, D] -> Q lists of (id, cosine), best first."""
        k = min(k, len(self))
        if k == 0:
            return [[] for _ in range(queries.shape[0])]
        top = torch.topk(self.scores(queries), k, dim=-1)                       # values, indices: [Q, k]
        return [[(self.id_of[j], s) for s, j in zip(vals.tolist(), idx.tolist())]
                for vals, idx in zip(top.values, top.indices)]

    def __len__(self) -> int:
        return len(self.slot_of)
