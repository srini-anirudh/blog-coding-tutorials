"""A mixture is a loss weighting, and the weighting leaks through token length (post section 4).

Three sources: a long-answer source standing in for chain-of-thought maths (13 supervised tokens), and two short
ones (5 tokens each). The nominal example mixture is 30/30/40. The script

  1. computes lambda_eff = p_i l_i / sum_j p_j l_j,
  2. measures the token share in 2,000 sampled batches,
  3. checks the exact identity grad(token-mean loss) = sum_i (T_i / T) grad(L_i) on one batch, and
  4. trains with token-mean and with per-example normalisation and compares held-out losses.

    python labs/mixing-data-without-losing-capabilities/scripts/02_token_weighting.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixlab import grads as G  # noqa: E402
from mixlab import mixtures as mx  # noqa: E402
from mixlab.model import make_model  # noqa: E402
from mixlab.runs import Report, n_procs, parallel_map  # noqa: E402
from mixlab.tasks import IGNORE  # noqa: E402
from mixlab.train import evaluate, make_pools, mixture_loss, sample_batch, scores, train  # noqa: E402

SOURCES = {"maths (long)": "long", "code (rev)": "rev", "chat (copy)": "copy"}
P = {"long": 0.3, "rev": 0.3, "copy": 0.4}
TR, EV = make_pools(tuple(P), n_train=20000, n_eval=1000)
STEPS = 400


def run(cfg):
    norm, weights, seed = cfg
    model = make_model(seed=seed)
    train(model, TR, weights, steps=STEPS, norm=norm, seed=seed)
    ev = evaluate(model, EV)
    return {t: ev[t]["loss"] for t in P}, scores(ev)


def main() -> None:
    rep = Report("02_token_weighting", workers=n_procs())
    lens = {t: TR[t].answer_tokens for t in P}
    lam = mx.token_weights([P[t] for t in P], [lens[t] for t in P])
    rep("1. formula: lambda_eff = p_i l_i / sum_j p_j l_j")
    rows = [[n, f"{P[t]:.2f}", f"{lens[t]:.1f}", f"{lam[k]:.3f}"] for k, (n, t) in enumerate(SOURCES.items())]

    gen = torch.Generator().manual_seed(0)
    tok = {t: 0 for t in P}
    for _ in range(2000):
        ids, lab, who = sample_batch(TR, P, 64, gen)
        n = (lab != IGNORE).sum(1)
        for name, c in zip(who, n.tolist()):
            tok[name] += c
    total = sum(tok.values())
    for r, t in zip(rows, P):
        r.append(f"{tok[t] / total:.3f}")
    rep.table(["source", "example share p", "mean supervised tokens", "lambda_eff (formula)", "token share, 2,000 batches"], rows)
    fix = mx.example_weights_for_token_target([P[t] for t in P], [lens[t] for t in P])
    rep("example shares that give a 30/30/40 token share: " + ", ".join(f"{t} {f:.3f}" for t, f in zip(P, fix)))
    rep("")

    rep("2. exact identity on one batch: grad(token mean) = sum_i (T_i/T) grad(token mean over source i)")
    model = make_model(seed=0)
    ids, lab, who = sample_batch(TR, P, 64, torch.Generator().manual_seed(1))
    full = G.flat_grad(model, mixture_loss(model, ids, lab, "token"))
    mask = lab != IGNORE
    combo = torch.zeros_like(full)
    for t in P:
        rows_t = torch.tensor([w == t for w in who])
        share = float(mask[rows_t].sum() / mask.sum())
        combo += share * G.flat_grad(model, mixture_loss(model, ids[rows_t], lab[rows_t], "token"))
    rep(f"   max |difference| = {float((full - combo).abs().max()):.2e}, |grad| = {float(full.norm()):.3f}")
    ex = G.flat_grad(model, mixture_loss(model, ids, lab, "example"))
    cosv = float(torch.nn.functional.cosine_similarity(full, ex, dim=0))
    rep(f"   cosine(token-mean gradient, per-example-mean gradient) on the same batch = {cosv:.3f}")
    rep("")

    rep(f"3. training consequence: {STEPS} steps, batch 64, 3 seeds each; held-out per-token loss (lower is better)")
    cfgs = [(norm, w, s) for norm, w in (("token", P), ("example", P), ("token", dict(zip(P, fix)))) for s in (0, 1, 2)]
    out = parallel_map(run, cfgs)
    labels = ["token-mean, p=30/30/40", "per-example, p=30/30/40", "token-mean, p corrected"]
    rows = []
    for k, lab_ in enumerate(labels):
        losses = [out[3 * k + s][0] for s in range(3)]
        accs = [out[3 * k + s][1] for s in range(3)]
        rows.append([lab_, *(f"{sum(l[t] for l in losses) / 3:.4f}" for t in P), *(f"{sum(a[t] for a in accs) / 3:.1f}" for t in P)])
        rep.data[lab_] = {"loss": losses, "acc": accs}
    rep.table(["setting", *(f"loss {n}" for n in SOURCES), *(f"acc% {n}" for n in SOURCES)], rows)
    rep.data.update({"lambda_eff": dict(zip(P, lam)), "token_share": {t: tok[t] / total for t in P}, "corrected_p": dict(zip(P, fix))})
    rep.save()


if __name__ == "__main__":
    main()
