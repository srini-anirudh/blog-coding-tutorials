"""How retrieval is measured: the post's worked examples, computed.

    python labs/searching-with-pixels/scripts/01_metrics.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixsearch import metrics as M  # noqa: E402
from pixsearch.common import Report  # noqa: E402

out = Report("01_metrics")
out("One relevant page, binary relevance. Where it sits changes each metric differently.")
out(f"{'rank':>4} {'R@1':>5} {'R@5':>5} {'R@10':>5} {'MRR':>6} {'nDCG@5':>7} {'1/log2(1+r)':>12}")
rows = []
for rank in (1, 2, 3, 4, 5, 6, 10):
    ranking = list(range(100, 120))
    ranking[rank - 1] = 0
    r = {"rank": rank, "R@1": M.recall_at_k(ranking, {0}, 1), "R@5": M.recall_at_k(ranking, {0}, 5),
         "R@10": M.recall_at_k(ranking, {0}, 10), "MRR": M.reciprocal_rank(ranking, {0}),
         "nDCG@5": M.ndcg_at_k(ranking, {0}, 5)}
    rows.append(r)
    ref = 1 / math.log2(1 + rank) if rank <= 5 else 0.0
    out(f"{rank:>4} {r['R@1']:>5.0f} {r['R@5']:>5.0f} {r['R@10']:>5.0f} {r['MRR']:>6.3f} {r['nDCG@5']:>7.3f} {ref:>12.3f}")

out("")
out("Graded relevance: three relevant pages with labels 3, 2, 1. Three rankings of the same pages.")
rel = {"A": 3, "B": 2, "C": 1}
for name, ranking in [("ideal  A B C x x", ["A", "B", "C", "x", "y"]),
                      ("swap   C B A x x", ["C", "B", "A", "x", "y"]),
                      ("late   x x A B C", ["x", "y", "A", "B", "C"])]:
    out(f"  {name}:  nDCG@5 = {M.ndcg_at_k(ranking, rel, 5):.3f}   MRR = {M.reciprocal_rank(ranking, set(rel)):.3f}"
        f"   R@1 = {M.recall_at_k(ranking, set(rel), 1):.0f}")
out("")
out("MRR and R@1 see only the first relevant page. nDCG also rewards putting the most relevant one first.")
out.save({"one_relevant": rows})
