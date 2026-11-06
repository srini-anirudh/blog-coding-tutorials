"""Part III: residual streams grow like sqrt(L), not g^L, and the identity term keeps the gradient alive.

    python labs/why-transformers-do-not-explode/scripts/03_residual_stream.py
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from stablelab.common import Report, seed_all  # noqa: E402

D = 256


def branch_weights(depth: int, rule: str, gen: torch.Generator) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One pre-norm MLP branch per layer: LN -> W1 (Kaiming) -> GELU -> W2. W2's scale depends on the rule."""
    out_std = {"none": 1.0, "1/sqrt(2L)": 1 / math.sqrt(2 * depth), "layerscale 1e-4": 1e-4}[rule]
    ws = []
    for _ in range(depth):
        w1 = torch.randn(4 * D, D, generator=gen, dtype=torch.float64) * math.sqrt(2 / D)
        w2 = torch.randn(D, 4 * D, generator=gen, dtype=torch.float64) / math.sqrt(4 * D) * out_std
        ws.append((w1, w2))
    return ws


def residual_norms(depth: int, rule: str, seed: int = 0) -> list[float]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(64, D, generator=g, dtype=torch.float64)
    norms = [x.norm(dim=-1).mean().item()]
    for w1, w2 in branch_weights(depth, rule, g):
        x = x + F.gelu(F.layer_norm(x, (D,)) @ w1.T) @ w2.T
        norms.append(x.norm(dim=-1).mean().item())
    return norms


def plain_norms(depth: int, gain: float, seed: int = 0) -> list[float]:
    """Plain linear stack with per-layer variance gain `gain`: the g^L comparison."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(64, D, generator=g, dtype=torch.float64)
    norms = [x.norm(dim=-1).mean().item()]
    for _ in range(depth):
        x = x @ (torch.randn(D, D, generator=g, dtype=torch.float64) * math.sqrt(gain / D)).T
        norms.append(x.norm(dim=-1).mean().item())
    return norms


def main() -> None:
    seed_all(0)
    r = Report("03_residual_stream")
    depth = 96
    r(f"1. Stream norm ||x_L|| / ||x_0|| after L pre-norm residual MLP blocks (width {D}, batch 64, float64),")
    r("   against a plain linear stack with variance gain 1.1 per layer (amplitude gain sqrt(1.1)).")
    curves = {rule: residual_norms(depth, rule) for rule in ("none", "1/sqrt(2L)", "layerscale 1e-4")}
    plain = plain_norms(depth, 1.1)
    rows = []
    for L in (6, 12, 24, 48, 96):
        rows.append([L] + [c[L] / c[0] for c in curves.values()] + [plain[L] / plain[0]])
    r.table(["L", "residual, none", "residual, 1/sqrt(2L)", "residual, LayerScale 1e-4", "plain, gain 1.1"], rows)
    c = np.array(curves["none"])
    v = (c[12] ** 2 - c[0] ** 2) / 12 / c[0] ** 2  # squared norm added per block, from the first 12 blocks only
    Ls = np.array([12, 24, 48, 96])
    slope = np.polyfit(np.log(Ls), np.log(c[Ls] ** 2 - c[0] ** 2), 1)[0]
    r(f"Plain residual: the first 12 blocks each add v = {v:.3f} * ||x_0||^2 of squared norm, so ||x_L||^2 ~ ||x_0||^2 (1 + L v).")
    r(f"Log-log slope of (||x_L||^2 - ||x_0||^2) against L: {slope:.3f} (1.0 means linear in L, so ||x_L|| ~ sqrt(L)).")
    r(f"Prediction for L = 96 from those 12 blocks: sqrt(1 + 96 v) = {math.sqrt(1 + 96 * v):.3f}; measured {c[96] / c[0]:.3f}.")
    r("Branch outputs are near-orthogonal to the stream, so they add in quadrature like a random walk.")
    r("Note: the 1/sqrt(2L) rule fixes the total added variance (about v/2) rather than letting it grow with L.")
    r()

    r("2. The identity term. Gradient norm at the bottom of an L-layer stack, unit gradient at the top.")
    rows = []
    for L in (8, 32, 128):
        g = torch.Generator().manual_seed(L)
        grad = torch.randn(D, generator=g, dtype=torch.float64)
        grad /= grad.norm()
        top = grad.clone()
        plain = grad.clone()
        for _ in range(L):
            plain = (torch.randn(D, D, generator=g, dtype=torch.float64) * math.sqrt(0.49 / D)).T @ plain
        res_dead = top.clone()  # every branch Jacobian is exactly zero: d x_{l+1} / d x_l = I
        rows.append([L, plain.norm().item(), 0.7**L, res_dead.norm().item()])
    r.table(["L", "plain, gain 0.7 per layer", "0.7^L", "residual, dead branches"], rows)
    r("With dead branches the residual gradient is the top gradient, unattenuated: the S = empty term is I.")
    r()

    r("3. The path ensemble. Expand prod_k (I + J_k) over subsets S for L = 10 linear branches with ||J|| ~ 0.4.")
    L, scale = 10, 0.4
    g = torch.Generator().manual_seed(1)
    Js = [torch.randn(64, 64, generator=g, dtype=torch.float64) * scale / math.sqrt(64) for _ in range(L)]
    full = torch.eye(64, dtype=torch.float64)
    for J in Js:
        full = (torch.eye(64, dtype=torch.float64) + J) @ full
    v0 = torch.randn(64, generator=g, dtype=torch.float64)
    by_len = np.zeros(L + 1)
    total = torch.zeros(64, dtype=torch.float64)
    for k in range(L + 1):
        part = torch.zeros(64, dtype=torch.float64)
        for S in itertools.combinations(range(L), k):
            vec = v0.clone()
            for i in S:  # ordered product J_{s_k} ... J_{s_1}
                vec = Js[i] @ vec
            part += vec
        by_len[k] = part.norm().item()
        total += part
    r(f"   Sum over all 2^{L} = {2**L} paths equals the product: max difference {(total - full @ v0).abs().max().item():.2e}")
    binom = np.array([math.comb(L, k) * scale**k for k in range(L + 1)])
    rows = [[k, math.comb(L, k), by_len[k] / by_len[0], binom[k]] for k in range(L + 1)]
    r.table(["path length k", "paths", "||sum of paths of length k|| / ||v||", "C(L,k) ||J||^k"], rows)
    mean_len = (np.arange(L + 1) * binom).sum() / binom.sum()
    r(f"The norm of each group falls far below the C(L,k) ||J||^k count: paths of the same length point in")
    r(f"near-random directions and partly cancel, so each group grows roughly like sqrt(C(L,k)) ||J||^k instead.")
    r(f"The post's binomial weighting C(L,k) ||J||^k gives mean path length L ||J|| / (1 + ||J||) = {mean_len:.2f};")
    w = by_len**2
    r(f"weighting by squared norms of the measured groups gives {(np.arange(L + 1) * w).sum() / w.sum():.2f}. Either way the")
    r(f"gradient is carried by paths far shorter than L = {L}.")
    r.save()


if __name__ == "__main__":
    main()
