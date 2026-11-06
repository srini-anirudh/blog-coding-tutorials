"""The residual stream at initialization, under four norm placements.

Post: "Variance accumulates ... ||x_l|| grows like sqrt(l)" and "put [the norm] on the identity path and you
break it, put it on the branch and you preserve it."

Part 1 traces the mean token norm of the stream through a 24-layer model, and fits ||x_l||^2 = a + b l.
Part 2 measures the gradient that reaches the first layer's weights, relative to the last layer's, at depth 6, 24, 48.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from organs.block import BlockConfig, Decoder
from organs.report import Report
from organs.train import batches, load_text

PLACEMENTS = ["post", "pre", "sandwich", "postbranch"]


def stream_norms(model: Decoder, x: torch.Tensor) -> list[float]:
    trace: list[torch.Tensor] = []
    with torch.no_grad():
        model(x, trace=trace)
    return [t.norm(dim=-1).mean().item() for t in trace]


def linear_fit_r2(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    x, y = torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)
    a = torch.stack([torch.ones_like(x), x], 1)
    coef = torch.linalg.lstsq(a, y[:, None]).solution.squeeze(1)
    resid = y - a @ coef
    r2 = 1 - resid.pow(2).sum() / (y - y.mean()).pow(2).sum()
    return coef[0].item(), coef[1].item(), r2.item()


def layer_grads(placement: str, layers: int, x: torch.Tensor, y: torch.Tensor, seed: int) -> tuple[float, float]:
    """||dL/dW|| of the FFN down-projection in the first and the last block, for one batch."""
    torch.manual_seed(seed)
    model = Decoder(BlockConfig(layers=layers, placement=placement, max_seq=x.shape[1]))
    F.cross_entropy(model(x).flatten(0, 1), y.flatten()).backward()
    g = [blk.ffn.down_proj.weight.grad.norm().item() for blk in model.blocks]
    return g[0], g[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    rep = Report("01_residual_stream")
    data = torch.frombuffer(bytearray(load_text()), dtype=torch.uint8)
    x, y = next(batches(data, 8, 128, torch.Generator().manual_seed(0)))

    rep(f"Part 1. Mean token norm of the residual stream, {args.layers} layers, d=128, at init (std 0.02), seed 0.")
    rows, norms = [], {}
    for p in PLACEMENTS:
        torch.manual_seed(0)
        norms[p] = stream_norms(Decoder(BlockConfig(layers=args.layers, placement=p)), x)
    picks = [0, 1, 2, 4, 8, 12, 16, 20, 24]
    picks = [i for i in picks if i <= args.layers]
    for i in picks:
        rows.append([i] + [f"{norms[p][i]:.3f}" for p in PLACEMENTS])
    rep.table(["layer"] + PLACEMENTS, rows)
    rep("")
    for p in ["pre", "sandwich", "postbranch"]:
        ls = list(range(1, args.layers + 1))
        a, b, r2 = linear_fit_r2(ls, [norms[p][i] ** 2 for i in ls])
        rep(f"{p:10s} fit ||x_l||^2 = {a:.3f} + {b:.3f} l   R^2 = {r2:.3f}   ||x_L|| / ||x_1|| = "
            f"{norms[p][-1] / norms[p][1]:.2f}")
    rep("post       every layer ends on a norm, so ||x_l|| is pinned at sqrt(d) = 11.3 (RMSNorm gain 1 at init).")
    rep("sandwich and postbranch normalize each branch output, so every write has norm sqrt(d) at init and")
    rep("||x_l||^2 grows by about d per write: exactly linear in l. The growth rate is set by the gain, which is")
    rep("why Trinity initializes that gain near 1/sqrt(L).")
    rep("")

    rep(f"Part 2. ||dL/dW|| of the FFN down-projection, first block / last block, one batch of Tiny Shakespeare,")
    rep(f"mean over {args.seeds} seeds. Xiong et al. (2020) predict that Post-LN's last-layer gradient does not shrink")
    rep("with depth, and Pre-LN's shrinks roughly as 1/sqrt(L). That is why Post-LN needs a warmup and Pre-LN does not.")
    rows = []
    for depth in (6, 24, 48):
        r = {p: [layer_grads(p, depth, x, y, s) for s in range(args.seeds)] for p in PLACEMENTS}
        rows.append([depth] + [f"{sum(a for a, _ in r[p]) / args.seeds:.4f} / {sum(b for _, b in r[p]) / args.seeds:.4f}"
                               for p in PLACEMENTS])
    rep.table(["depth"] + PLACEMENTS, rows)
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    main()
