"""Three ways to score a match: bi-encoder, late interaction (MaxSim) and cross-encoder, trained from
scratch on the same synthetic binding task, then graded on the same held-out corpus.

    python labs/searching-with-pixels/scripts/02_three_scorers.py [--steps 1500]

Writes results/02_three_scorers.{txt,json}; checkpoints go to runs/ (not in git) for scripts 03 and 05.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
import torch  # noqa: E402

from pixsearch import pages as P  # noqa: E402
from pixsearch import retrieve as R  # noqa: E402
from pixsearch import train as T  # noqa: E402
from pixsearch.common import Report, fast_mode, save_model  # noqa: E402
from pixsearch.models import count_params  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--pages", type=int, default=2000, help="held-out corpus size (page, twin) pairs x 2")
ap.add_argument("--queries", type=int, default=500)
ap.add_argument("--cross-queries", type=int, default=100, help="queries for the full cross-encoder scan")
args = ap.parse_args()
if fast_mode():
    args.steps, args.pages, args.queries, args.cross_queries = 30, 200, 50, 10

torch.manual_seed(0)
out = Report("02_three_scorers")
test, queries, rel = C.heldout(args.pages, args.queries)
src = queries.source
out(f"Held-out corpus: {len(test)} pages ({len(test) // 2} page/twin pairs), {len(queries)} binding queries,")
out(f"on average {sum(map(len, rel)) / len(rel):.2f} relevant pages per query. Training: {args.steps} steps of 64 fresh pages.")
out("")

configs = [("bi", 16), ("bi", 64), ("bi", 256), ("late", 16), ("cross", 0)]
results = {}
for kind, dim in configs:
    name = f"{kind}{dim}" if dim else kind
    out(f"== {name}")
    model = T.build(kind, dim or 64)
    t0 = time.time()
    if kind == "cross":
        hist = T.train_cross(model, steps=args.steps, log=out)
    else:
        hist = T.train_dual(model, steps=args.steps, log=out)
    train_s = time.time() - t0
    model.eval()
    save_model(model, name, {"kind": kind, "dim": dim or 64, "steps": args.steps})
    if kind == "cross":
        q = P.Queries(queries.tokens[:args.cross_queries], queries.combos[:args.cross_queries],
                      queries.source[:args.cross_queries])
        secs = R.time_queries(lambda: R.cross_score_all(model, q.tokens, test.images), reps=1)
        scores = R.cross_score_all(model, q.tokens, test.images)
        m = R.evaluate_scores(scores, q, rel[:args.cross_queries])
        index_s, nq, passes = 0.0, len(q), len(test)
    else:
        index_s = R.time_queries(lambda: R.index_pages(model, test.images), reps=1)
        idx = R.index_pages(model, test.images)
        secs = R.time_queries(lambda: R.score_all(model, queries.tokens, idx))
        scores = R.score_all(model, queries.tokens, idx)
        m = R.evaluate_scores(scores, queries, rel)
        nq, passes = len(queries), 1
    rec = {"metrics": m, "params": count_params(model), "train_s": round(train_s, 1),
           "stored_floats_per_page": R.stored_floats_per_page(model), "index_s": round(index_s, 3),
           "ms_per_query": round(1000 * secs / nq, 3), "encoder_passes_per_query": passes,
           "queries_scored": nq, "history": hist}
    results[name] = rec
    out(f"  {m}")
    out("")

out("Summary (held-out corpus; percent):")
hdr = f"{'model':<8}{'stored floats/page':>19}{'passes/query':>14}{'ms/query':>10}{'P@1':>7}{'nDCG@5':>8}{'R@10':>7}{'beats twin':>11}"
out(hdr)
for name, r in results.items():
    m = r["metrics"]
    out(f"{name:<8}{r['stored_floats_per_page']:>19}{r['encoder_passes_per_query']:>14}{r['ms_per_query']:>10.3f}"
        f"{m['P@1']:>7.1f}{m['nDCG@5']:>8.1f}{m['R@10']:>7.1f}{m['beats_twin']:>11.1f}")
out(f"(cross is scored on the first {args.cross_queries} queries only: it needs one forward pass per page.)")

# A MaxSim map, as in the post's Figure 5: where each query word finds its best patch.
from pixsearch.common import load_model  # noqa: E402

late, _ = load_model("late16", lambda meta: T.build(meta["kind"], meta["dim"]))
with torch.no_grad():
    qv = late.encode_queries(queries.tokens[:1])[0]
    dv = late.encode_pages(test.images[src[:1]])[0]
sim = qv @ dv.T
out("")
out(f"MaxSim map for query \"{P.describe_query(queries.tokens[0])}\" on its page:")
it = test.items[src[0]]
glyph = {int(c): f"{P.COLOR_NAMES[int(k)][:3]}{int(s)}" for s, k, c in it.tolist()}
out("  page layout (colour + shape id per cell, . = blank):")
for r in range(P.GRID):
    out("   " + " ".join(f"{glyph.get(r * P.GRID + c, '.'):>6}" for c in range(P.GRID)))
words = P.describe_query(queries.tokens[0]).split()
for t, w in enumerate(words):
    best = int(sim[t].argmax())
    out(f"  token {t} '{w}': best patch = cell {best} ({glyph.get(best, 'blank')}), similarity {sim[t].max():.3f}")
results["maxsim_map"] = {"query": " ".join(words), "best_cells": sim.argmax(1).tolist(),
                         "best_sims": [round(x, 4) for x in sim.max(1).values.tolist()]}
out.save(results)
