"""Part IV: where the normalizer goes. LN's Jacobian, per-layer gradients at init, and warmup.

    python labs/why-transformers-do-not-explode/scripts/04_norm_placement.py --steps 400     # about 15 min on 8 CPU threads
    python labs/why-transformers-do-not-explode/scripts/04_norm_placement.py --steps 100   # quick look
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from stablelab.common import Report, seed_all  # noqa: E402
from stablelab.model import CharData, Config, TinyTransformer, evaluate, load_corpus, train  # noqa: E402


def layer_grad_norms(model: TinyTransformer, data: CharData, seed: int = 0) -> list[float]:
    """Gradient norm of each block's weights for one batch at initialization."""
    x, y = data.batch(32, torch.Generator().manual_seed(seed))
    model.zero_grad()
    model.loss(x, y).backward()
    return [math.sqrt(sum(p.grad.pow(2).sum().item() for p in blk.parameters())) for blk in model.blocks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3])
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args()
    seed_all(0)
    r = Report("04_norm_placement")

    r("1. ||d LN(x) / dx|| (spectral norm, no affine) for x = r * u, d = 256. Xiong et al.: O(sqrt(d) / ||x||).")
    d = 256
    u = torch.randn(d, dtype=torch.float64)
    u = (u - u.mean()) / (u - u.mean()).norm()  # centred, unit norm
    rows = []
    for radius in (1.0, 4.0, 16.0, 64.0, 256.0):
        jac = torch.autograd.functional.jacobian(lambda z: F.layer_norm(z, (d,), eps=0.0), u * radius)
        s = torch.linalg.matrix_norm(jac, ord=2).item()
        rows.append([radius, s, math.sqrt(d) / radius, s * radius / math.sqrt(d)])
    r.table(["||x||", "||dLN/dx||", "sqrt(d)/||x||", "ratio"], rows)
    r("Exact for centred x: the Jacobian is (sqrt(d)/||x||) times a projection, so the ratio is 1.")
    r()

    data = CharData(load_corpus(), seq=64)
    L = args.layers
    r(f"2. Gradient norm of each block's weights at initialization, {L} layers, d = 128, one Tiny Shakespeare batch.")
    variants = [("post", "default"), ("post", "xavier"), ("pre", "default"), ("pre", "xavier")]
    norms = {}
    for norm, init in variants:
        torch.manual_seed(0)
        norms[(norm, init)] = layer_grad_norms(TinyTransformer(Config(vocab=data.vocab, layers=L, norm=norm,
                                                                      init=init)), data)
    rows = [[i + 1] + [norms[v][i] for v in variants] for i in range(L)]
    r.table(["block"] + [f"{n}-LN {i}" for n, i in variants], rows)
    for v in variants:
        r(f"{v[0]}-LN, {v[1]} init: top block / bottom block gradient norm = {norms[v][-1] / norms[v][0]:.3f}")
    r("With Xavier init (Xiong et al.'s setting) Post-LN puts more gradient on the top blocks, because every LN on")
    r("the identity path divides the gradient by the norm of x + F(x). With N(0, 0.02) init the branches start near")
    r("zero, x + F(x) is about x, and the Post-LN profile is flat. Pre-LN puts more gradient on the bottom blocks")
    r("under either init: the imbalance the post attributes to Shleifer et al.")
    r()

    r(f"3. Training, {L} layers, d = 128, batch 32 x 64 chars, AdamW (0.9, 0.95), clip 1.0, {args.steps} steps.")
    r(f"   Warmup is linear over {args.warmup} steps, then constant. Validation loss on held-out text; 3.0 nats is")
    r("   roughly a unigram model, 2.5 a bigram-level model.")
    rows = []
    for lr in args.lrs:
        for norm, init in (("post", "default"), ("post", "xavier"), ("pre", "default")):
            for warmup in (0, args.warmup):
                torch.manual_seed(0)
                model = TinyTransformer(Config(vocab=data.vocab, layers=L, norm=norm, init=init))
                hist = train(model, data, args.steps, lr, warmup=warmup, log_every=50)
                early = hist["loss"][min(2, len(hist["loss"]) - 1)]
                val = float("nan") if hist["diverged"] else evaluate(model, data)
                rows.append([lr, norm, init, warmup, early, hist["loss"][-1], val, max(hist["grad_norm"])])
                r(f"   done: lr {lr} {norm}-LN {init} warmup {warmup}: val {val:.3f}")
    r.table(["lr", "norm", "init", "warmup", "train loss @ step 100", "final train loss", "val loss", "max grad norm"], rows)
    r.save()


if __name__ == "__main__":
    main()
