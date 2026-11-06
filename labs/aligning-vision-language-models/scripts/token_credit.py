"""Fine-grained feedback: where the DPO gradient goes (post Figure 7), and whether it matters for training.

    python labs/aligning-vision-language-models/scripts/token_credit.py

Part 1. For pairs with one wrong colour, compute the per-token push on the parameters,
    v_t = grad log pi(y_w,t | ctx) - grad log pi(y_l,t | ctx'),
for a minimal edit (same claims, same order, one word changed) and for a rewritten winner (same correct facts in a
new order). Shared-prefix tokens cancel exactly in the minimal pair. We also measure how much of the whole pair
gradient points along the edited claim, with plain DPO and with Dense DPO weights (gamma = 5).
Part 2. Token weights from an image contrast (TPO-style): |log p(y_t | I) - log p(y_t | noised I)|, by token type.
Part 3. Train DPO on 500 minimal pairs versus 500 rewritten pairs and compare.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from groundpref.model import pad, seq_logprob, token_logprobs  # noqa: E402
from groundpref.prefopt import VARIANTS, fact_accuracy, train_pref  # noqa: E402
from groundpref.pairs import Pairs, build_pairs  # noqa: E402
from groundpref.report import Log  # noqa: E402
from groundpref.train import describe_eval, eval_scenes, load_sft, set_threads  # noqa: E402
from groundpref.world import (COLOURS, COUNTS, OBJECTS, PROMPT_LEN, SEP, STYLE, TOK, VOCAB,  # noqa: E402
                              encode, hallucinate, render, sample_scene)


def make_pair(rng, gen):
    """One scene, the hallucinated loser, a minimally edited winner and a rewritten winner."""
    while True:
        s = sample_scene(rng)
        if len(s.facts) >= 3:
            break
    good = s.claims()
    rng.shuffle(good)
    bad = hallucinate(s, good, "colour", rng)
    i = next(k for k, (a, b) in enumerate(zip(good, bad)) if a != b)
    rewritten = good[:]
    while rewritten == good:
        rng.shuffle(rewritten)
    return s, render(s, gen), encode(good), encode(bad), encode(rewritten), good[i].obj


def claim_span(ids: list[int], obj: str) -> list[bool]:
    """Mask over target positions (ids[1:]) covering the claim about `obj`: count colour object ;"""
    j = ids.index(TOK[obj])
    m = [False] * (len(ids) - 1)
    for t in range(j - 2, j + 2):
        m[t - 1] = True
    return m


def flat_grad(model, scalar):
    gs = torch.autograd.grad(scalar, [p for p in model.parameters()], retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                      for g, p in zip(gs, model.parameters())])


def per_token_grads(model, img, ids):
    lp = token_logprobs(model, img[None], torch.tensor([ids]))[0]
    return [flat_grad(model, lp[t]) for t in range(PROMPT_LEN - 1, len(ids) - 1)]


def pair_gradient(model, img, w, l, ww=None, wl=None):
    """Gradient of the DPO margin log pi(w) - log pi(l), optionally with Dense DPO token weights."""
    pw = seq_logprob(model, img[None], torch.tensor([w]), None if ww is None else ww[None])
    pl = seq_logprob(model, img[None], torch.tensor([l]), None if wl is None else wl[None])
    return flat_grad(model, (pw - pl).sum())


def dense(ids, span, gamma):
    w = torch.ones(len(ids) - 1)
    w[torch.tensor(span)] = gamma
    return w


def part1(model, log, n_pairs):
    rng, gen = random.Random(5), torch.Generator().manual_seed(5)
    pos_rows, share = [], {"minimal": [], "rewritten": [], "minimal_dense": [], "rewritten_dense": []}
    for _ in range(n_pairs):
        s, img, good, bad, rew, obj = make_pair(rng, gen)
        gw, gl = per_token_grads(model, img, good), per_token_grads(model, img, bad)
        edit = next(t for t in range(len(good)) if good[t] != bad[t])
        pos_rows.append([((a - b).norm().item(), t + PROMPT_LEN - edit) for t, (a, b) in enumerate(zip(gw, gl))])
        span_l = claim_span(bad, obj)
        for name, w in (("minimal", good), ("rewritten", rew)):
            span_w = claim_span(w, obj)
            full = pair_gradient(model, img, w, bad)
            # the same margin restricted to the edited claim's tokens
            lw = token_logprobs(model, img[None], torch.tensor([w]))[0]
            ll = token_logprobs(model, img[None], torch.tensor([bad]))[0]
            edit_only = flat_grad(model, lw[torch.tensor(span_w)].sum() - ll[torch.tensor(span_l)].sum())
            share[name].append(torch.cosine_similarity(full, edit_only, 0).item())
            fd = pair_gradient(model, img, w, bad, dense(w, span_w, 5.0), dense(bad, span_l, 5.0))
            share[name + "_dense"].append(torch.cosine_similarity(fd, edit_only, 0).item())
    log("Part 1a. Minimal pairs: norm of the net per-token push v_t, by position relative to the edit")
    by = {}
    for row in pos_rows:
        for v, rel in row:
            by.setdefault(max(-4, min(rel, 6)), []).append(v)
    out_pos = {}
    for rel in sorted(by):
        lab = "<=-4" if rel == -4 else ("+6+" if rel == 6 else f"{rel:+d}")
        out_pos[lab] = {"mean": statistics.mean(by[rel]), "max": max(by[rel]), "n": len(by[rel])}
        log(f"  position {lab:>5s} (0 = edited word)  mean |v_t| = {statistics.mean(by[rel]):.4f}"
            f"   max = {max(by[rel]):.4f}   tokens = {len(by[rel])}")
    log("\nPart 1b. Cosine between the whole-pair gradient and the gradient of the edited claim alone")
    out_share = {}
    for k, v in share.items():
        out_share[k] = statistics.mean(v)
        log(f"  {k:16s} {statistics.mean(v):.3f}  (sd {statistics.stdev(v):.3f}, {len(v)} pairs)")
    return {"push_by_position": out_pos, "cosine_with_edit_gradient": out_share}


@torch.no_grad()
def part2(model, log, n=400):
    """TPO-style token weights: how much each token's log-prob depends on the image."""
    rng, gen = random.Random(6), torch.Generator().manual_seed(6)
    groups = {"count": set(TOK[w] for w in COUNTS), "colour": set(TOK[w] for w in COLOURS),
              "object": set(TOK[w] for w in OBJECTS), "separator ;": {SEP}, "style": set(TOK[w] for w in STYLE),
              "<eos>": {TOK["<eos>"]}}
    acc = {g: [] for g in groups}
    for _ in range(n):
        s = sample_scene(rng)
        claims = s.claims()
        rng.shuffle(claims)
        ids = torch.tensor([encode(claims, [rng.choice(STYLE) if rng.random() < 0.3 else None for _ in claims])])
        img = render(s, gen)[None]
        noisy = img + 3.0 * torch.randn(img.shape, generator=gen)
        d = (token_logprobs(model, img, ids) - token_logprobs(model, noisy, ids)).abs()[0]
        for t in range(PROMPT_LEN - 1, ids.shape[1] - 1):
            tok = ids[0, t + 1].item()
            for g, ws in groups.items():
                if tok in ws:
                    acc[g].append(d[t].item())
    log("\nPart 2. Image-contrast token weight |log p(y_t|I) - log p(y_t|noised I)|, noise sd 3.0, by token type")
    out = {}
    for g, v in acc.items():
        out[g] = statistics.mean(v)
        log(f"  {g:12s} {statistics.mean(v):.3f}   ({len(v)} tokens; e.g. '{VOCAB[next(iter(groups[g]))]}')")
    return out


def pairs_of(kind: str, n: int, seed: int) -> Pairs:
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    p = Pairs([], [], [], [], [])
    for _ in range(n):
        s, img, good, bad, rew, _ = make_pair(rng, gen)
        p.scenes.append(s)
        p.imgs.append(img)
        p.yw.append(good if kind == "minimal" else rew)
        p.yl.append(bad)
        p.kinds.append("colour")
    p.imgs = torch.stack(p.imgs)
    return p


def part3(model, log, seeds, n):
    log(f"\nPart 3. DPO on {n} colour-error pairs: minimal-edit winners versus rewritten winners")
    scenes, imgs = eval_scenes(600, seed=3000)
    fact = build_pairs(1500, seed=2000, flavour="fact", kinds=("colour",))
    out = {}
    for kind in ("minimal", "rewritten"):
        rows = []
        for seed in seeds:
            pol, _ = train_pref(model, pairs_of(kind, n, 100 + seed), VARIANTS["dpo"], steps=100, seed=seed,
                                log=lambda *a: None)
            r = {**describe_eval(pol, scenes, imgs), **fact_accuracy(pol, model, fact)}
            rows.append(r)
        m = {k: statistics.mean(r[k] for r in rows) for k in ("colour_err", "chair_i", "recall",
                                                              "policy_pref_acc", "implicit_reward_acc")}
        out[kind] = {"mean": m, "runs": rows}
        log(f"  {kind:10s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()) + f"   ({len(seeds)} seeds)")
    base = {**describe_eval(model, scenes, imgs), **fact_accuracy(model, model, fact)}
    log(f"  {'sft':10s} colour_err={base['colour_err']:.3f}  chair_i={base['chair_i']:.3f}  "
        f"recall={base['recall']:.3f}  policy_pref_acc={base['policy_pref_acc']:.3f}")
    out["sft"] = base
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=60, help="pairs for the per-token gradient analysis")
    ap.add_argument("--train-pairs", type=int, default=500)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    set_threads()
    log = Log("token_credit")
    model = load_sft(log)
    res = {"part1": part1(model, log, args.pairs), "part2": part2(model, log),
           "part3": part3(model, log, args.seeds, args.train_pairs)}
    log.save({"args": vars(args), **res})


if __name__ == "__main__":
    main()
