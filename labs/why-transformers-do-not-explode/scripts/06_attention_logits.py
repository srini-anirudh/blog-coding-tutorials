"""Part VI: logit growth, softmax saturation, entropy collapse, and what QK-norm and soft-capping bound.

    python labs/why-transformers-do-not-explode/scripts/06_attention_logits.py                 # about 12 min, 8 threads
    python labs/why-transformers-do-not-explode/scripts/06_attention_logits.py --no-train      # parts 1-3 only, seconds
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from stablelab.common import Report, seed_all  # noqa: E402
from stablelab.model import CharData, Config, TinyTransformer, evaluate, load_corpus, train  # noqa: E402


def softmax_stats(z: torch.Tensor) -> dict:
    p = z.softmax(-1)
    jac = torch.diag(p) - torch.outer(p, p)
    return {"entropy": -(p * p.clamp_min(1e-300).log()).sum().item(), "max_p": p.max().item(),
            "max_jac": (p * (1 - p)).max().item(), "jac_norm": torch.linalg.matrix_norm(jac, ord=2).item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2, 3e-2])
    args = ap.parse_args()
    seed_all(0)
    r = Report("06_attention_logits" if not args.no_train else "06_attention_logits_no_train")

    r("1. Sixteen logits z = s * z0 (z0 ~ N(0,1), fixed), float64, as the growth factor s sweeps.")
    z0 = torch.randn(16, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    uniform = softmax_stats(torch.zeros(16, dtype=torch.float64))
    rows, sweep = [], []
    for s in torch.logspace(-1, 3, 401, dtype=torch.float64).tolist():
        sweep.append((s, softmax_stats(s * z0)))
    for s in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
        st = softmax_stats(s * z0)
        rows.append([s, st["entropy"], st["max_p"], st["max_jac"], uniform["max_jac"] / max(st["max_jac"], 1e-300)])
    r.table(["s", "entropy (nats)", "max p", "max p(1-p)", "uniform / this"], rows)
    at_005 = next(st for s, st in sweep if st["entropy"] < 0.05)
    at_1e3 = next(st for s, st in sweep if st["max_p"] > 1 - 1e-3)
    r(f"Uniform: entropy ln 16 = {math.log(16):.3f} nats, max p(1-p) = {uniform['max_jac']:.4f}.")
    r(f"First s with entropy < 0.05 nats: max p = {at_005['max_p']:.4f}, max p(1-p) = {at_005['max_jac']:.4f},"
      f" {uniform['max_jac'] / at_005['max_jac']:.1f}x below uniform (the post says about six times).")
    r(f"First s with max p within 1e-3 of 1: max p(1-p) = {at_1e3['max_jac']:.2e},"
      f" {uniform['max_jac'] / at_1e3['max_jac']:.0f}x below uniform; entropy {at_1e3['entropy']:.4f} nats.")
    r("Entropy falls first; the Jacobian only collapses once one probability is almost exactly 1. At moderate s the")
    r("largest p(1-p) is above the uniform value (it peaks at 0.25 when some p = 1/2), so growth first sharpens a head")
    r("without weakening its gradient at all.")
    r()

    r("2. Softmax Jacobian p_i(delta_ij - p_j) against autograd, 16 random logits:")
    z = torch.randn(16, dtype=torch.float64) * 3
    p = z.softmax(-1)
    auto = torch.autograd.functional.jacobian(lambda t: t.softmax(-1), z)
    r(f"   max |formula - autograd| = {(torch.diag(p) - torch.outer(p, p) - auto).abs().max().item():.2e}")
    r()

    r("3. What each mitigation bounds. One head, d = 64, d_h = 16, 32 tokens; W_Q and W_K multiplied by a growth")
    r("   factor g (the logit grows like g^2 without a fix). QK-norm is LayerNorm with unit gain; soft-cap c = 50.")
    x = torch.randn(32, 64, dtype=torch.float64)
    wq, wk = (torch.randn(16, 64, dtype=torch.float64) / 8 for _ in range(2))
    ln = lambda t: torch.nn.functional.layer_norm(t, (16,))  # noqa: E731
    rows = []
    for g in (1, 4, 16, 64):
        q, k = x @ (g * wq).T, x @ (g * wk).T
        raw = q @ k.T / 4
        normed = ln(q) @ ln(k).T / 4
        capped = 50 * torch.tanh(raw / 50)
        rows.append([g, raw.abs().max().item(), normed.abs().max().item(), capped.abs().max().item()])
    r.table(["g", "max |z| raw", "max |z| QK-norm", "max |z| soft-cap 50"], rows)
    r(f"QK-norm bound: |z| <= ||LN(q)|| ||LN(k)|| / sqrt(d_h) = d_h / sqrt(d_h) = sqrt(d_h) = {math.sqrt(16):.0f}.")
    r("Nuance: with a learnable LayerNorm gain the bound scales with the gains, so QK-norm bounds the logits only")
    r("as long as the gains themselves stay bounded.")
    r()
    if args.no_train:
        r.save()
        return

    data = CharData(load_corpus(), seq=64)
    r(f"4. A small-scale proxy in the spirit of Wortsman et al.: 4 layers, d = 128, 4 heads, Pre-LN, AdamW (0.9, 0.95),")
    r(f"   warmup 100, no clipping, {args.steps} steps, batch 32 x 64 chars. Max |logit| and min entropy over all heads.")
    rows = []
    for lr in args.lrs:
        for qk in (False, True):
            torch.manual_seed(0)
            model = TinyTransformer(Config(vocab=data.vocab, layers=4, qk_norm=qk))
            hist = train(model, data, args.steps, lr, warmup=100, clip=None, log_every=50)
            val = float("nan") if hist["diverged"] else evaluate(model, data)
            peak = max(range(len(hist["max_logit"])), key=lambda i: hist["max_logit"][i])
            rows.append([lr, "on" if qk else "off", hist["max_logit"][-1], min(hist["entropy"]),
                         hist["step"][peak], max(hist["loss"][2:]), val, "yes" if hist["diverged"] else "no"])
            print(f"  lr {lr} qk_norm {qk}: final max logit {hist['max_logit'][-1]:.1f}, val {val:.3f}", flush=True)
    r.table(["lr", "QK-norm", "final max |z|", "min entropy", "step of peak |z|", "worst train loss after step 100",
             "val loss", "NaN"], rows)
    r("Read the max-logit column against the LR: this is the logged quantity the post says gives days of warning.")
    r.save()


if __name__ == "__main__":
    main()
