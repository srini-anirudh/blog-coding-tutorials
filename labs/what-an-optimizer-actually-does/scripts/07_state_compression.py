"""Spending less state: is Adam's second moment rank-1, and what does 8-bit storage lose?

Train TinyLLM with AdamW for 200 steps, then take its real m and v and ask:
  1. Adafactor: how close is each matrix's v to the rank-1 rebuild R C^T / sum(R), and how close is
     the update m / sqrt(V_hat) to Adam's m / sqrt(v)?
  2. 8-bit: the error of m, v and the update under one scale per tensor vs one per 2048-value block,
     with a linear and a log code. Is v more sensitive than m? What does a single outlier do?

Run: python labs/what-an-optimizer-actually-does/scripts/07_state_compression.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optlab import Adam, BlockQuantized, factored_v
from optlab.common import Report
from optlab.train import train

EPS = 1e-8


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def update(m, v):
    return m / (v.sqrt() + EPS)


def main():
    rep = Report("07_state_compression")
    r = train(lambda m: Adam(m.parameters(), lr=3e-3, weight_decay=0.1), steps=200)
    rep(f"AdamW, 200 steps, final train loss {r['train_loss']:.3f}")
    model, opt = r["model"], r["opt"]
    t = next(iter(opt.state.values()))["t"]
    b1, b2 = opt.betas
    mats = [(n, opt.state[p]) for n, p in model.named_parameters() if p.ndim == 2]

    rep("\n## 1. Adafactor's rank-1 second moment on real Adam state")
    rep(f"{'tensor':<32} {'shape':>11} {'storage':>16} {'|V-Vhat|/|V|':>13} {'cos(update)':>12}")
    cosines = []
    for name, s in mats:
        m, v = s["m"] / (1 - b1**t), s["v"] / (1 - b2**t)
        vhat = factored_v(v.mean(1), v.mean(0))
        c = cos(update(m, vhat), update(m, v))
        cosines.append(c)
        rows, cols = v.shape
        rep(f"{name:<32} {str(tuple(v.shape)):>11} {rows * cols:>7} -> {rows + cols:<6} {rel(vhat, v):>13.3f} {c:>12.3f}")
    rep(f"mean cosine between Adafactor-style and Adam updates: {sum(cosines) / len(cosines):.3f}")

    rep("\n## 2. 8-bit state: relative error, pooled over all 2-D tensors")
    m_all = torch.cat([s["m"].reshape(-1) for _, s in mats])
    v_all = torch.cat([s["v"].reshape(-1) for _, s in mats])
    u_ref = update(m_all, v_all)
    rep(f"{'scheme':<22} {'err m':>8} {'err v':>8} {'err u (m only)':>15} {'err u (v only)':>15} {'err u (both)':>13}")
    errs = {}
    for label, block, code in (("per-tensor, linear", None, "linear"), ("block 2048, linear", 2048, "linear"),
                               ("per-tensor, log", None, "log"), ("block 2048, log", 2048, "log")):
        qm = BlockQuantized(m_all, block, code, signed=True).dequantize()
        qv = BlockQuantized(v_all, block, code, signed=False).dequantize()
        e = [rel(qm, m_all), rel(qv, v_all), rel(update(qm, v_all), u_ref), rel(update(m_all, qv), u_ref),
             rel(update(qm, qv), u_ref)]
        errs[label] = e
        rep(f"{label:<22} " + " ".join(f"{x:>{w}.3f}" for x, w in zip(e, (8, 8, 15, 15, 13))))
    zero = (BlockQuantized(v_all, 2048, "linear", signed=False).dequantize() == 0) & (v_all > 0)
    rep(f"block 2048 linear: {zero.float().mean():.1%} of nonzero v entries round to 0, so m/(sqrt(v)+eps) "
        f"becomes m/eps there")

    rep("\n## 3. One outlier: multiply a single entry of m by 1000")
    m_out = m_all.clone()
    m_out[0] = m_all.abs().max() * 1000
    for label, block in (("per-tensor", None), ("block 2048", 2048)):
        q = BlockQuantized(m_out, block, "log", signed=True).dequantize()
        lost = ((q == 0) & (m_out != 0)).float().mean().item()
        rep(f"{label:<11} log code: {lost:.1%} of m entries become 0; error on the rest {rel(q[1:], m_out[1:]):.3f}")
    rep.save({"adafactor_cosines": cosines, "quant_errors": errs})


if __name__ == "__main__":
    main()
