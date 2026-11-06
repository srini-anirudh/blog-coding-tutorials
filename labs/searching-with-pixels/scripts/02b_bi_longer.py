"""Is the bi-encoder's failure on twins a capacity limit or a training limit?

A 256-d vector has room for all 128 (shape, colour) glyphs, so in principle one pooled vector can count
how many query glyphs a page holds. This trains bi256 four times longer than script 02 and grades it on
the same held-out queries.

    python labs/searching-with-pixels/scripts/02b_bi_longer.py [--steps 6000]
"""

import argparse

import _common as C
import torch

from pixsearch import retrieve as R
from pixsearch import train as T
from pixsearch.common import Report, fast_mode

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--dim", type=int, default=256)
args = ap.parse_args()
if fast_mode():
    args.steps = 20

out = Report("02b_bi_longer")
test, queries, rel = C.heldout()
torch.manual_seed(0)
model = T.build("bi", args.dim)
hist = T.train_dual(model, steps=args.steps, log=out, log_every=max(1, args.steps // 8))
model.eval()
m = R.evaluate_scores(R.score_all(model, queries.tokens, R.index_pages(model, test.images)), queries, rel)
out(f"bi{args.dim} after {args.steps} steps: {m}")
out.save({"steps": args.steps, "dim": args.dim, "metrics": m, "history": hist})
