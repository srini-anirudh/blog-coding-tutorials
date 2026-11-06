"""Shared setup for the scripts: import path, the held-out corpus, and loading trained models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os  # noqa: E402

import torch  # noqa: E402

# The models are tiny. On a busy shared CPU, extra threads made them much slower, so default to one.
torch.set_num_threads(int(os.environ.get("PIXSEARCH_THREADS", "1")))

from pixsearch import pages as P  # noqa: E402
from pixsearch import train as T  # noqa: E402
from pixsearch.common import fast_mode, load_model  # noqa: E402


def heldout(pages: int = 2000, queries: int = 500):
    """The same held-out corpus and queries as script 02 (seed 1000)."""
    if fast_mode():
        pages, queries = 200, 50
    g = torch.Generator().manual_seed(1000)
    test = P.make_corpus(pages, g)
    src = torch.arange(0, pages, max(1, pages // queries))[:queries]
    q = P.make_queries(test, g, pages=src)
    return test, q, P.relevant_sets(test, q)


def load(name: str):
    model, _ = load_model(name, lambda meta: T.build(meta["kind"], meta["dim"]))
    return model


def subset(q: P.Queries, n: int) -> P.Queries:
    return P.Queries(q.tokens[:n], q.combos[:n], q.source[:n])
