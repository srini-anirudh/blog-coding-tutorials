"""Sections 01 and 07: attention is permutation equivariant, and the causal mask quietly breaks that.

1. Permute the input of one attention layer. Without a mask the output permutes with it, exactly.
2. With a causal mask, or with RoPE, it does not.
3. A uniform head over a causal prefix returns 1/(i+1) of a "first token" signal: position from counting.
4. Train a tiny NoPE model to output each token's own index, with a causal and with a bidirectional mask.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posenc import tasks, tinylm  # noqa: E402
from posenc.report import Report, threads_from_env  # noqa: E402
from posenc.rope import cos_sin, inv_freq  # noqa: E402


def equivariance_gap(attn, x, causal, rope=False) -> float:
    perm = torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(0))
    kw = {}
    if rope:
        cos, sin = cos_sin(torch.arange(x.shape[1]), inv_freq(attn.dh))
        kw["rope"] = (cos[None], sin[None])
    with torch.no_grad():
        a = attn(x[:, perm], causal=causal, **kw)
        b = attn(x, causal=causal, **kw)[:, perm]
    return float((a - b).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--length", type=int, default=64)
    args = ap.parse_args()
    threads_from_env()
    torch.manual_seed(0)
    rep = Report("01_symmetry")

    rep.print("\n1-2. max |f(Px) - P f(x)| for one attention layer, d=32, 4 heads, 12 tokens")
    attn = tinylm.Attention(32, 4)
    x = torch.randn(1, 12, 32)
    rows = [
        ["no mask, no position", equivariance_gap(attn, x, causal=False)],
        ["causal mask, no position", equivariance_gap(attn, x, causal=True)],
        ["no mask, RoPE", equivariance_gap(attn, x, causal=False, rope=True)],
    ]
    rep.table(["setting", "max gap"], rows)
    rep.data["equivariance_gap"] = {r[0]: r[1] for r in rows}

    rep.print("\n3. A uniform causal head. Token 0 carries value e; every other value is orthogonal to e.")
    n, d = 64, 16
    e = torch.zeros(d)
    e[0] = 1.0
    values = torch.randn(n, d)
    values[:, 0] = 0.0
    values[0] = e
    uniform = torch.ones(n, n).tril()
    uniform = uniform / uniform.sum(-1, keepdim=True)
    out = uniform @ values
    rows = [[i, float(out[i, 0]), 1 / (i + 1)] for i in (0, 1, 3, 7, 15, 31, 63)]
    rep.table(["position i", "component on e", "1/(i+1)"], rows)

    rep.print(f"\n4. Predict your own position (0..{args.length - 1}) from random tokens. 2 layers, d=64.")
    rep.print(f"   {args.steps} AdamW steps, batch 64. Chance accuracy is {1 / args.length:.3f}.")
    batch = tasks.make("position", length=args.length, vocab=64)
    results = {}
    for name, pe, causal in [("NoPE, causal", "none", True), ("NoPE, bidirectional", "none", False),
                             ("learned table, causal", "learned", True)]:
        torch.manual_seed(0)
        model = tinylm.TinyLM(64, n_out=args.length, d=64, pe=pe, max_len=args.length, causal=causal)
        tinylm.train(model, batch, args.steps, lr=3e-3)
        gen = torch.Generator().manual_seed(99)
        tokens, targets, _ = batch(gen)
        with torch.no_grad():
            pred = model(tokens).argmax(-1)
        err = (pred - targets).abs().float()
        results[name] = {
            "accuracy": float((err == 0).float().mean()),
            "mean_abs_error": float(err.mean()),
            "accuracy_pos_0_7": float((err[:, :8] == 0).float().mean()),
            "mean_abs_error_pos_48_63": float(err[:, 48:].mean()),
        }
    rep.table(["model", "exact acc", "mean |error|", "acc, pos 0-7", "mean |error|, pos 48-63"],
              [[k, *v.values()] for k, v in results.items()])
    rep.data["position_probe"] = results
    rep.save()


if __name__ == "__main__":
    main()
