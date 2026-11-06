"""Utilisation of padding and packing on the real TinyStories document lengths.

    python c05-pretraining/benchmarks/bench_packing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pretrain.packing import (best_fit_packing_utilization, dynamic_padding_utilization,  # noqa: E402
                              greedy_packing_utilization, padding_utilization)
from pretrain.utils import resolve  # noqa: E402


def main() -> None:
    lens = np.load(resolve("data_cache/tinystories/train_doc_lens.npy"))
    print(f"{len(lens):,} documents, {lens.sum():,} tokens, length p10 {np.percentile(lens, 10):.0f} "
          f"median {np.median(lens):.0f} p90 {np.percentile(lens, 90):.0f} max {lens.max()}")
    print(f"{'L':>5} {'strategy':<24}{'utilisation':>12}{'tokens cut':>12}{'multi-doc windows':>19}{'split docs':>12}")
    for L in (128, 256, 512):
        rows = [("pad to L", padding_utilization(lens, L)),
                ("pad to batch max (B=32)", dynamic_padding_utilization(lens, L, 32)),
                ("greedy packing", greedy_packing_utilization(lens, L)),
                ("best-fit packing", best_fit_packing_utilization(lens, L))]
        for name, r in rows:
            md = f"{100 * r['multi_doc_frac']:.1f}%" if "multi_doc_frac" in r else "-"
            sd = f"{100 * r['split_docs_frac']:.1f}%" if "split_docs_frac" in r else "-"
            print(f"{L:>5} {name:<24}{100 * r['utilization']:>11.1f}%{100 * r['truncated_frac']:>11.1f}%{md:>19}{sd:>12}")


if __name__ == "__main__":
    main()
