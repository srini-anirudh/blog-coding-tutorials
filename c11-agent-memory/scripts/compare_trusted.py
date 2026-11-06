"""Compare the index and the packer against trusted implementations.

    python c11-agent-memory/scripts/compare_trusted.py > c11-agent-memory/results/trusted.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentmem.bench.hardware import hardware_label, use_threads  # noqa: E402
from agentmem.bench.trusted import compare_index, compare_packer  # noqa: E402


def main() -> None:
    use_threads(8)
    i = compare_index()
    print(f"TensorIndex.search vs. F.cosine_similarity + topk ({hardware_label()})")
    print(f"  N={i['n']}, D={i['dim']}, k={i['k']}: same ranking on {i['same_ranking']}/{i['queries']} queries, "
          f"largest score difference {i['max_score_difference']:.2e}")
    p = compare_packer()
    print(f"\nContextBudgeter.pack vs. exact knapsack by brute force, {p['trials']} random sets of {p['blocks']} blocks")
    print(f"  total value, greedy / optimal: mean {p['mean_ratio']:.3f}, worst {p['worst_ratio']:.3f}")
    print(f"  greedy found the optimum on {p['optimal_found']}/{p['trials']} sets")


if __name__ == "__main__":
    main()
