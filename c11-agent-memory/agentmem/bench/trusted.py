"""Check our two core algorithms against implementations we did not write.

The index against `torch.nn.functional.cosine_similarity` plus `torch.topk`, and the
greedy packer against an exact brute-force knapsack. The greedy packer is meant to be
close to optimal, not equal to it, so the second check reports how close.
"""

from __future__ import annotations

import itertools
import random

import torch

from ..budget import Block, ContextBudgeter, Section
from ..index_fast import TensorIndex


def compare_index(n: int = 2000, dim: int = 384, queries: int = 50, k: int = 10, seed: int = 0) -> dict[str, float]:
    """Same ids in the same order, and the largest score difference, over random unit vectors."""
    g = torch.Generator().manual_seed(seed)
    vectors = torch.nn.functional.normalize(torch.randn(n, dim, generator=g), dim=-1)  # [N, D]
    index = TensorIndex(dim)
    index.add([str(i) for i in range(n)], vectors)
    same, worst = 0, 0.0
    for q in torch.randn(queries, dim, generator=g):                                   # q: [D]
        ours = index.search(torch.nn.functional.normalize(q, dim=-1), k)
        cos = torch.nn.functional.cosine_similarity(q.unsqueeze(0), vectors, dim=-1)   # [N]
        top = torch.topk(cos, k)
        same += [i for i, _ in ours] == [str(i) for i in top.indices.tolist()]
        worst = max(worst, max(abs(s - t) for (_, s), t in zip(ours, top.values.tolist())))
    return {"n": n, "dim": dim, "k": k, "queries": queries, "same_ranking": same, "max_score_difference": worst}


def best_value(blocks: list[Block], budgeter: ContextBudgeter, budget: int) -> float:
    """The exact optimum by trying all 2^n subsets, with the packer's own cost accounting."""
    header = budgeter.header_cost(blocks[0].section)
    best = 0.0
    for size in range(1, len(blocks) + 1):
        for subset in itertools.combinations(blocks, size):
            if header + sum(budgeter.cost(b) for b in subset) <= budget:
                best = max(best, sum(b.value for b in subset))
    return best


def compare_packer(trials: int = 300, n: int = 12, seed: int = 0) -> dict[str, float]:
    """Greedy total value / optimal total value on random block sets. Compression is off,
    because the exact solver only knows whole blocks."""
    rng = random.Random(seed)
    budgeter = ContextBudgeter([Section("memory", "memory")])
    ratios = []
    for _ in range(trials):
        blocks = [Block("memory", " ".join(["word"] * rng.randint(4, 60)), value=rng.random(), compressible=False, id=str(i))
                  for i in range(n)]
        budget = rng.randint(40, 200)
        packed = budgeter.pack(blocks, budget)
        assert packed.total <= budget
        optimal = best_value(blocks, budgeter, budget)
        ratios.append(sum(b.value for b in packed.blocks) / optimal if optimal else 1.0)
    return {"trials": trials, "blocks": n, "mean_ratio": sum(ratios) / trials, "worst_ratio": min(ratios),
            "optimal_found": sum(r > 1 - 1e-9 for r in ratios)}
