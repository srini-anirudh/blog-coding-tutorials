"""Cascades: a cheap bi-encoder retrieves the top k, a late-interaction or cross-encoder model reorders
them. The reranker can only reorder what the first stage found, so Recall@k of stage one caps it.

    python labs/searching-with-pixels/scripts/03_cascade.py      (needs the checkpoints from script 02)
"""

import time

import _common as C
import torch

from pixsearch import retrieve as R
from pixsearch.common import Report
from pixsearch.metrics import evaluate, recall_at_k

out = Report("03_cascade")
test, queries, rel = C.heldout()
bi, late, cross = C.load("bi64"), C.load("late16"), C.load("cross")

with torch.no_grad():
    bi_scores = R.score_all(bi, queries.tokens, R.index_pages(bi, test.images))
    late_index = R.index_pages(late, test.images)
first = bi_scores.topk(100, dim=1).indices
base = R.evaluate_scores(bi_scores, queries, rel)
out(f"Stage one alone (bi64 over all {len(test)} pages): P@1 {base['P@1']:.1f}  nDCG@5 {base['nDCG@5']:.1f}")
out("")
out(f"{'k':>4}{'stage-1 R@k':>12}{'| late rerank P@1':>18}{'nDCG@5':>8}{'| cross rerank P@1':>19}{'nDCG@5':>8}"
    f"{'cross passes':>13}{'cross ms/query':>15}")
rows = []
for k in (1, 2, 5, 10, 20, 50, 100):
    cand = first[:, :k]
    rk = 100 * sum(recall_at_k(c, r, k) for c, r in zip(cand.tolist(), rel)) / len(rel)
    late_rank, cross_rank = [], []
    t0 = time.perf_counter()
    for i in range(len(queries)):
        c = cand[i]
        with torch.no_grad():
            s = cross.score_pairs(queries.tokens[i], test.images[c])
        cross_rank.append(c[s.argsort(descending=True)].tolist())
    cross_ms = 1000 * (time.perf_counter() - t0) / len(queries)
    for i in range(len(queries)):
        c = cand[i]
        with torch.no_grad():
            s = late.score(late.encode_queries(queries.tokens[i:i + 1]), late_index[c])[0]
        late_rank.append(c[s.argsort(descending=True)].tolist())
    ml, mc = evaluate(late_rank, rel), evaluate(cross_rank, rel)
    rows.append({"k": k, "stage1_recall": round(rk, 2), "late_P@1": round(ml["P@1"], 2),
                 "late_nDCG@5": round(ml["nDCG@5"], 2), "cross_P@1": round(mc["P@1"], 2),
                 "cross_nDCG@5": round(mc["nDCG@5"], 2), "cross_passes": k, "cross_ms_per_query": round(cross_ms, 2)})
    out(f"{k:>4}{rk:>12.1f}{ml['P@1']:>18.1f}{ml['nDCG@5']:>8.1f}{mc['P@1']:>19.1f}{mc['nDCG@5']:>8.1f}"
        f"{k:>13}{cross_ms:>15.2f}")
out("(nDCG@5 after reranking counts only pages inside the top k, so k < 5 caps it.)")

out("")
out("The post's Figure 8 cost model, as arithmetic (not a measurement): 1,000 visual tokens per page,")
out("a 30-token query, corpus vectors computed at indexing time.")
model_rows = []
for corpus, k, n in [(40_000, 50, 5), (40_000, 10, 3), (1_000_000, 50, 5)]:
    stage1 = 30
    rerank = k * (1000 + 30)
    read = n * 1000 + 30
    read_all = corpus * 1000
    model_rows.append({"corpus": corpus, "k": k, "n": n, "stage1_tokens": stage1, "rerank_tokens": rerank,
                       "read_tokens": read, "read_everything_tokens": read_all})
    out(f"  corpus {corpus:>9,}  k={k:<3} n={n}:  stage 1 {stage1:>3} tokens, rerank {rerank:>7,}, read {read:>6,};"
        f"  reading every page instead: {read_all:,}")
out.save({"first_stage": base, "cascade": rows, "token_cost_model": model_rows})
