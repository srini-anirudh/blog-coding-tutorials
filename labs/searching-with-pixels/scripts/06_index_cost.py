"""What a multi-vector index costs, and how much quality survives each way of shrinking it.

First the post's storage arithmetic (ColPali, BGE-M3, BM25 numbers are the paper's, only multiplied
here). Then the trained late16 model from script 02: merge or prune its 16 patch vectors per page, and
quantise them, and re-grade the same held-out queries.

    python labs/searching-with-pixels/scripts/06_index_cost.py      (needs the checkpoints from script 02)
"""

import _common as C
import torch

from pixsearch import index as I
from pixsearch import retrieve as R
from pixsearch.common import Report

out = Report("06_index_cost")
result: dict = {}

colpali = I.storage_bytes(1, 1030, 128, 2)
single = I.storage_bytes(1, 1, 2048, 2)
out("Storage arithmetic (inputs from the post):")
out(f"  ColPali: 1,030 vectors x 128 dims x 2 bytes = {colpali:,.0f} bytes = {colpali / 1024:.1f} KiB per page")
out(f"  one 2,048-d fp16 vector = {single:,.0f} bytes; ratio {colpali / single:.1f}x")
for pages in (1_000_000, 100_000_000):
    out(f"  {pages:>11,} pages: ColPali {I.storage_bytes(pages, 1030, 128, 2) / 1e9:,.1f} GB, "
        f"single vector {I.storage_bytes(pages, 1, 2048, 2) / 1e9:,.1f} GB")
for f in (3,):
    v = -(-1030 // f)
    out(f"  pool factor {f}: {v} vectors per page, {I.storage_bytes(1, v, 128, 2) / 1024:.1f} KiB "
        f"({100 * (1 - v / 1030):.1f}% of vectors removed)")
out(f"  binary (1 bit per dim): {I.storage_bytes(1, 1030, 128, 1 / 8) / 1024:.1f} KiB per page (16x smaller than fp16)")
result["arithmetic"] = {"colpali_bytes": colpali, "single_2048_fp16_bytes": single, "ratio": colpali / single}

test, queries, rel = C.heldout()
late, bi = C.load("late16"), C.load("bi64")
with torch.no_grad():
    raw = torch.cat([late.encode_pages(test.images[i:i + 512], normalize=False) for i in range(0, len(test), 512)])
    qv = late.encode_queries(queries.tokens)
norms = raw.norm(dim=-1)
full = torch.nn.functional.normalize(raw, dim=-1)
dim = full.shape[-1]


def grade(scores):
    return R.evaluate_scores(scores, queries, rel)


base = grade(late.score(qv, full))
with torch.no_grad():
    bi_m = grade(R.score_all(bi, queries.tokens, R.index_pages(bi, test.images)))
out("")
out(f"Our late16 index: {len(test)} pages x 16 vectors x {dim} dims. Baseline nDCG@5 {base['nDCG@5']:.1f}, "
    f"P@1 {base['P@1']:.1f}. (bi64, one 64-d vector per page: nDCG@5 {bi_m['nDCG@5']:.1f}.)")
out("")
out(f"{'variant':<34}{'vectors/page':>13}{'bytes/page':>11}{'nDCG@5':>8}{'retained':>9}{'P@1':>7}")
rows = []


def row(name, vecs, bytes_per_dim, m):
    b = vecs * dim * bytes_per_dim
    ret = 100 * m["nDCG@5"] / base["nDCG@5"]
    rows.append({"variant": name, "vectors_per_page": vecs, "bytes_per_page": b, **m, "retained_pct": round(ret, 1)})
    out(f"{name:<34}{vecs:>13}{b:>11.0f}{m['nDCG@5']:>8.1f}{ret:>8.1f}%{m['P@1']:>7.1f}")


row("fp32, all patches", 16, 4, base)
row("fp16", 16, 2, grade(late.score(qv, I.to_fp16(full))))
row("int8 (per-vector scale)", 16, 1, grade(late.score(qv, I.to_int8(full))))
row("binary (sign only)", 16, 1 / 8, grade(late.score(qv, I.to_binary(full))))
for k in (20, 50):
    row(f"binary, rescore top {k} in fp32", 16, 1 / 8, grade(I.binary_then_rescore(qv, full, k)))
for f in (2, 3, 4, 8):
    keep = -(-16 // f)
    row(f"merge, pool factor {f}", keep, 4, grade(late.score(qv, I.merge_pool(full, f))))
    row(f"prune by norm, keep {keep}", keep, 4, grade(late.score(qv, I.prune(full, f, "norm", norms))))
    row(f"prune at random, keep {keep}", keep, 4, grade(late.score(qv, I.prune(full, f, "random"))))
out("")
out("Binary rows store 1 bit per dimension; the rescoring rows also need the fp32 vectors of the short list.")
result["late16"] = rows
result["bi64"] = bi_m
out.save(result)
