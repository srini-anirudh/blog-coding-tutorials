"""Contrastive training for the toy retrievers (section "The contrastive loss is a retrieval loss").

The loss is InfoNCE: each query must pick its page out of every page in the batch, and (for the dual
models) each page must pick its query, CLIP-style. Pages that truly answer a query but are not its
labelled page can be masked out of the softmax using the exact labels (`mask_false_negatives`).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

from . import pages as P
from .models import BiEncoder, CrossEncoder, DualModel, LateInteraction


def info_nce(logits: torch.Tensor, ignore: torch.Tensor | None = None) -> torch.Tensor:
    """logits: Q x N, the positive of query i is column i. ignore: Q x N bool, columns dropped from
    the softmax (never the positive)."""
    if ignore is not None:
        logits = logits.masked_fill(ignore, float("-inf"))
    return F.cross_entropy(logits, torch.arange(logits.shape[0]))


def info_nce_grad(t: torch.Tensor, v: torch.Tensor, pos: int, tau: float) -> torch.Tensor:
    """The post's closed form for one query: dL/dt = -(1/tau) [v_pos - sum_j p_j v_j],
    p = softmax(<t, v_j> / tau). t: dim, v: N x dim (unnormalised dot products)."""
    p = torch.softmax(v @ t / tau, dim=0)
    return -(v[pos] - p @ v) / tau


def false_negative_mask(cand_combos: torch.Tensor, q_combos: torch.Tensor) -> torch.Tensor:
    """Q x N bool: candidate n holds both glyphs of query q, but n is not q's labelled page (n != q)."""
    rel = cand_combos[:, q_combos[:, 0]] & cand_combos[:, q_combos[:, 1]]  # N x Q
    rel = rel.T.clone()
    idx = torch.arange(q_combos.shape[0])
    rel[idx, idx] = False
    return rel


def _schedule(opt: torch.optim.Optimizer, step: int, steps: int, lr: float, warmup: int) -> None:
    if step < warmup:
        f = (step + 1) / warmup
    else:
        f = 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
    for gr in opt.param_groups:
        gr["lr"] = lr * f


# A miner returns extra negative pages for a batch: (images M x 3 x 32 x 32, combos M x N_COMBOS, which
# query each one belongs to M) or None. Used by the hard-negative experiment.
Miner = Callable[[int, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]


def dual_scores(model: DualModel, tokens: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    q = model.encode_queries(tokens)
    d = model.encode_pages(images)
    return model.score(q, d)


def train_dual(model: DualModel, steps: int = 1500, batch: int = 64, lr: float = 2e-3, seed: int = 0,
               twins: bool = True, mask_false_negatives: bool = True, warmup: int = 100,
               sampler: Callable[[int, torch.Generator], tuple[P.Corpus, P.Queries, torch.Tensor]] | None = None,
               miner: Miner | None = None, log_every: int = 250, log: Callable[[str], None] = print) -> list[dict]:
    """Train a BiEncoder or LateInteraction model with symmetric InfoNCE.

    By default each step draws a fresh batch of `batch` pages (with twins in the batch when twins=True).
    `sampler(step, g)` replaces that with pages from a fixed pool; it also returns the pool ids.
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    hist: list[dict] = []
    t0 = time.time()
    for step in range(steps):
        _schedule(opt, step, steps, lr, warmup)
        if sampler is None:
            corpus, queries = P.training_batch(batch, g, twins=twins)
            ids = torch.arange(batch)
        else:
            corpus, queries, ids = sampler(step, g)
        images, cand = corpus.images, corpus.combos
        extra = miner(step, ids) if miner is not None else None
        n_pages = images.shape[0]
        if extra is not None:
            ex_img, ex_combos, owner = extra
            images = torch.cat([images, ex_img])
            cand = torch.cat([cand, ex_combos])
        logits = model.scale() * dual_scores(model, queries.tokens, images)    # B x (B + M)
        ignore = false_negative_mask(cand, queries.combos) if mask_false_negatives else None
        if extra is not None and ignore is not None:
            # Mined negatives are kept or filtered by the miner itself, not here.
            ignore[:, n_pages:] = False
        loss_q = info_nce(logits, ignore)
        loss_p = info_nce(logits[:, :n_pages].T, None if ignore is None else ignore[:, :n_pages].T)
        loss = 0.5 * (loss_q + loss_p)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                acc = (logits[:, :n_pages].argmax(1) == torch.arange(n_pages)).float().mean().item()
            rec = {"step": step + 1, "loss": round(loss.item(), 4), "batch_acc": round(acc, 3),
                   "tau": round(1 / model.scale().item(), 4), "time_s": round(time.time() - t0, 1)}
            hist.append(rec)
            log(f"  step {rec['step']:5d}  loss {rec['loss']:.3f}  in-batch acc {acc:.3f}  "
                f"tau {rec['tau']:.4f}  {rec['time_s']:.0f} s")
    return hist


def train_cross(model: CrossEncoder, steps: int = 1500, batch: int = 64, group: int = 8, lr: float = 1e-3,
                seed: int = 0, warmup: int = 100, log_every: int = 250,
                log: Callable[[str], None] = print) -> list[dict]:
    """Listwise training: each query scores a group of `group` pages from its batch (its page, its twin,
    and random others) and softmax cross-entropy picks its page. Relevant non-labelled pages are masked."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    hist: list[dict] = []
    t0 = time.time()
    for step in range(steps):
        _schedule(opt, step, steps, lr, warmup)
        corpus, queries = P.training_batch(batch, g, twins=True)
        b = batch
        idx = torch.arange(b)
        others = torch.stack([torch.randperm(b, generator=g)[:group + 2] for _ in range(b)])
        cols = [idx, idx ^ 1]
        for k in range(group + 2):
            if len(cols) == group:
                break
            cols.append(others[:, k])
        cand = torch.stack(cols, dim=1)                       # B x G, column 0 is the positive
        # A random column can repeat the positive; it is masked below, as are other relevant pages.
        tok = queries.tokens[:, None].expand(-1, group, -1).reshape(-1, P.QUERY_LEN)
        img = corpus.images[cand.reshape(-1)]
        logits = model(tok, img).view(b, group)
        has = corpus.combos
        rel = has[cand, queries.combos[:, 0:1]] & has[cand, queries.combos[:, 1:2]]
        dup = cand == idx[:, None]
        ignore = (rel | dup)
        ignore[:, 0] = False
        loss = F.cross_entropy(logits.masked_fill(ignore, float("-inf")), torch.zeros(b, dtype=torch.long))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % log_every == 0 or step == steps - 1:
            acc = (logits.masked_fill(ignore, float("-inf")).argmax(1) == 0).float().mean().item()
            rec = {"step": step + 1, "loss": round(loss.item(), 4), "group_acc": round(acc, 3),
                   "time_s": round(time.time() - t0, 1)}
            hist.append(rec)
            log(f"  step {rec['step']:5d}  loss {rec['loss']:.3f}  group acc {acc:.3f}  {rec['time_s']:.0f} s")
    return hist


def build(kind: str, dim: int = 64) -> torch.nn.Module:
    if kind == "bi":
        return BiEncoder(dim)
    if kind == "late":
        return LateInteraction(dim)
    if kind == "cross":
        return CrossEncoder()
    raise ValueError(kind)
