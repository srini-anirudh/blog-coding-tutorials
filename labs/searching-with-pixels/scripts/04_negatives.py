"""Why negatives decide what is learned.

Part A checks the post's InfoNCE gradient, dL/dt = -(1/tau)[v+ - sum_j p_j v_j], against autograd, and
measures on the trained late16 model how little weight the easy negatives of a real corpus get.
Part B trains the same late-interaction model (16-d) four ways on one fixed pool of pages, 30% of which have an exact
clone (a second page that answers the same queries):
  random       random batches from the pool: twins rarely meet in a batch
  twins        each batch holds pages with their twins (hard negatives by construction)
  mined        every 100 steps, re-index the pool with the model and add each query's top-scoring
               non-relevant page as an extra negative (ANCE-style, filtered with the exact labels)
  mined-unfiltered   the same, but only the labelled page is skipped, so clones become false negatives

    python labs/searching-with-pixels/scripts/04_negatives.py [--steps 1000]   (Part A needs script 02)
"""

import argparse

import _common as C
import torch

from pixsearch import pages as P
from pixsearch import retrieve as R
from pixsearch import train as T
from pixsearch.common import Report, fast_mode

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--pool", type=int, default=4096)
ap.add_argument("--refresh", type=int, default=100)
args = ap.parse_args()
if fast_mode():
    args.steps, args.pool, args.refresh = 20, 256, 10

out = Report("04_negatives")
result: dict = {}

# ---- Part A
g = torch.Generator().manual_seed(0)
t = torch.randn(32, generator=g, requires_grad=True)
v = torch.randn(64, 32, generator=g)
loss = torch.nn.functional.cross_entropy((v @ t / 0.05)[None], torch.tensor([0]))
loss.backward()
err = (t.grad - T.info_nce_grad(t.detach(), v, 0, 0.05)).abs().max().item()
out(f"Part A. Closed-form InfoNCE gradient against autograd: max difference {err:.2e}")

test, queries, rel = C.heldout()
bi = C.load("late16")
with torch.no_grad():
    s = R.score_all(bi, queries.tokens, R.index_pages(bi, test.images))
    tau = 1 / bi.scale().item()
p = torch.softmax(s / tau, dim=1)
mask = torch.zeros_like(p, dtype=torch.bool)
for i, r in enumerate(rel):
    mask[i, list(r)] = True
neg = p.masked_fill(mask, 0)
neg = neg / neg.sum(1, keepdim=True)        # share of the negatives' total weight
top = neg.sort(dim=1, descending=True).values
n_neg = (~mask).sum(1).float()
share1, share10 = top[:, 0].mean().item(), top[:, :10].sum(1).mean().item()
tiny = ((neg < 1e-6) & ~mask).sum(1).float().div(n_neg).mean().item()
out(f"  Trained late16 (tau = {tau:.4f}) over {len(test)} pages: of the push-away weight on negatives,")
out(f"  the single hardest negative gets {100 * share1:.1f}% and the top 10 get {100 * share10:.1f}% on average;")
out(f"  {100 * tiny:.1f}% of negatives get a weight below 1e-6 (they contribute almost no gradient).")
twin_rank = []
for i, src in enumerate(queries.source.tolist()):
    order = neg[i].argsort(descending=True)
    twin_rank.append(int((order == (src ^ 1)).nonzero()) + 1)
out(f"  The source page's twin is the hardest negative for {100 * sum(r == 1 for r in twin_rank) / len(twin_rank):.1f}%"
    f" of queries (median rank {sorted(twin_rank)[len(twin_rank) // 2]}).")
result["partA"] = {"grad_max_diff": err, "tau": tau, "hardest_share": share1, "top10_share": share10,
                   "frac_below_1e-6": tiny}

# ---- Part B
out("")
gp = torch.Generator().manual_seed(7)
pool = P.make_corpus(args.pool, gp, kind="twins", clone_frac=0.3)
pool_q = P.make_queries(pool, gp, binding=True)
pool_rel = P.relevant_sets(pool, pool_q)
pool_combos = pool.combos
n_clone = sum(any(j != i and bool((pool_combos[j] == pool_combos[i]).all()) for j in r) for i, r in enumerate(pool_rel))
out(f"Part B. Pool of {len(pool)} pages. A query has {sum(map(len, pool_rel)) / len(pool):.1f} relevant pages on average "
    f"(two glyphs often co-occur by chance); {n_clone} queries ({100 * n_clone / len(pool):.0f}%) also have an exact clone "
    f"of their page. {args.steps} steps, batch 64.")
# Twins are generated next to each other unless a clone was inserted between them; find them by their shapes.
shape_key = [tuple(sorted(it[:, 0].tolist())) for it in pool.items]
by_key: dict = {}
for i, k in enumerate(shape_key):
    by_key.setdefault(k, []).append(i)


def sampler_random(step, g):
    ids = torch.randperm(len(pool), generator=g)[:64]
    return P.Corpus(pool.items[ids], pool.images[ids]), P.Queries(pool_q.tokens[ids], pool_q.combos[ids], ids), ids


def sampler_twins(step, g):
    seeds = torch.randperm(len(pool), generator=g)[:32].tolist()
    ids = []
    for i in seeds:
        mates = [j for j in by_key[shape_key[i]] if j != i]
        ids += [i, mates[int(torch.randint(len(mates), (1,), generator=g))] if mates else i]
    ids = torch.tensor(ids)
    return P.Corpus(pool.items[ids], pool.images[ids]), P.Queries(pool_q.tokens[ids], pool_q.combos[ids], ids), ids


def make_miner(model, filtered: bool, stats: dict):
    hard = torch.zeros(len(pool), dtype=torch.long)

    def miner(step, ids):
        if step % args.refresh == 0:
            model.eval()
            with torch.no_grad():
                sc = R.score_all(model, pool_q.tokens, R.index_pages(model, pool.images))
            model.train()
            sc[torch.arange(len(pool)), torch.arange(len(pool))] = -float("inf")
            if filtered:
                for i, r in enumerate(pool_rel):
                    sc[i, list(r)] = -float("inf")
            hard.copy_(sc.argmax(1))
            false = sum(int(hard[i]) in pool_rel[i] for i in range(len(pool)))
            stats.setdefault("false_negative_rate", []).append(round(100 * false / len(pool), 2))
        h = hard[ids]
        return pool.images[h], pool_combos[h], torch.arange(len(ids))

    return miner


variants = {}
for name in ("random", "twins", "mined", "mined-unfiltered"):
    out(f"== {name}")
    torch.manual_seed(0)
    model = T.build("late", 16)
    stats: dict = {}
    sampler = sampler_twins if name == "twins" else sampler_random
    miner = make_miner(model, name == "mined", stats) if name.startswith("mined") else None
    T.train_dual(model, steps=args.steps, sampler=sampler, miner=miner, log=out, log_every=max(1, args.steps // 4))
    model.eval()
    with torch.no_grad():
        sc = R.score_all(model, queries.tokens, R.index_pages(model, test.images))
    m = R.evaluate_scores(sc, queries, rel)
    variants[name] = {"metrics": m, **stats}
    extra = f"  mined negatives that were really relevant: {stats['false_negative_rate']}%" if stats else ""
    out(f"  held-out: P@1 {m['P@1']:.1f}  nDCG@5 {m['nDCG@5']:.1f}  beats twin {m['beats_twin']:.1f}{extra}")

out("")
out(f"{'variant':<18}{'P@1':>7}{'nDCG@5':>8}{'R@10':>7}{'beats twin':>11}")
for name, r in variants.items():
    m = r["metrics"]
    out(f"{name:<18}{m['P@1']:>7.1f}{m['nDCG@5']:>8.1f}{m['R@10']:>7.1f}{m['beats_twin']:>11.1f}")
result["partB"] = variants
out.save(result)
