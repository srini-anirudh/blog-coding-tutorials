"""Reference vector index: a Python list and a loop.

It is O(N * D) Python operations per query and exists to be obviously correct.
`index_fast.TensorIndex` must return the same ids and scores (tests check this).
"""

from __future__ import annotations

import torch


class NaiveIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self.vectors: list[list[float]] = []

    def add(self, ids: list[str], vectors: torch.Tensor) -> None:
        """vectors: [N, D], already L2-normalised."""
        for i, v in zip(ids, vectors.tolist()):
            self.ids.append(i)
            self.vectors.append(v)

    def remove(self, id_: str) -> None:
        j = self.ids.index(id_)
        del self.ids[j], self.vectors[j]

    def search(self, query: torch.Tensor, k: int) -> list[tuple[str, float]]:
        """query: [D]. Returns the k (id, cosine) pairs with the highest cosine."""
        q = query.tolist()
        scored = []
        for id_, v in zip(self.ids, self.vectors):
            dot = 0.0
            for a, b in zip(q, v):
                dot += a * b
            scored.append((id_, dot))
        scored.sort(key=lambda p: -p[1])
        return scored[:k]

    def __len__(self) -> int:
        return len(self.ids)
