"""The FFN learned to gate: the 8/3 accounting, and the gate as a multiplication primitive.

Post sections: "The 8/3 accounting" and "Two changes, not one" ("That multiplicative interaction is something a
two-matrix MLP has to approximate with depth and width; here it is a primitive").

Part 2 trains a two-matrix GELU FFN and a SwiGLU FFN with matched parameter counts on two toy targets: products of
input pairs, and a sum of one-input functions as a control. This is a toy regression, not language modelling: it
shows what the gate makes easy, not the "GLU variants improve the Transformer" result, which the post itself
calls unexplained.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from organs.arith import swiglu_parity_width
from organs.block import GeluFFN
from organs.report import Report
from tinyllm.mlp import SwiGLU

D, PAIRS = 16, 8


def targets(x: torch.Tensor, kind: str) -> torch.Tensor:
    a, b = x[:, 0::2], x[:, 1::2]  # (N, 8) each
    return a * b if kind == "product" else torch.sin(a) + torch.cos(b)


def fit(model: torch.nn.Module, head: torch.nn.Linear, kind: str, seed: int, steps: int = 3000) -> float:
    g = torch.Generator().manual_seed(seed)
    xtr, xte = torch.randn(8192, D, generator=g), torch.randn(4096, D, generator=g)
    ytr, yte = targets(xtr, kind), targets(xte, kind)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        idx = torch.randint(0, 8192, (256,), generator=g)
        loss = F.mse_loss(head(model(xtr[idx])), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return (F.mse_loss(head(model(xte)), yte) / yte.var()).item()


def main() -> None:
    rep = Report("05_ffn_gate")
    rep("1. The 8/3 accounting: 3 d d_ff = 8 d^2 gives d_ff = 8d/3, rounded up to a multiple of 256.")
    rows = []
    for name, d, shipped in [("Llama 2 7B", 4096, 11008), ("Llama 3 8B", 4096, 14336), ("Olmo 3 32B", 5120, 27648)]:
        exact, rounded = swiglu_parity_width(d)
        rows.append([name, d, f"{exact:,.1f}", f"{rounded:,}", f"{shipped:,}", f"{shipped / d:.3f}",
                     f"{3 * d * shipped / (8 * d * d):.3f}"])
    rep.table(["model", "d", "8d/3", "next x256", "shipped d_ff", "d_ff / d", "params vs 4d GELU FFN"], rows)
    rep("Llama 2 ships the parity width. Llama 3 and Olmo 3 spend more than parity on the FFN.")
    rep("")

    width_gelu = 4 * D
    width_glu = round(8 * D / 3)
    gelu, glu = GeluFFN(D, width_gelu), SwiGLU(D, width_glu)
    n_gelu = sum(p.numel() for p in gelu.parameters())
    n_glu = sum(p.numel() for p in glu.parameters())
    rep(f"2. Toy regression, d = {D}, {PAIRS} outputs through a shared linear read-out.")
    rep(f"   GELU FFN: 2 matrices, width {width_gelu}, {n_gelu} parameters. "
        f"SwiGLU: 3 matrices, width {width_glu}, {n_glu} parameters.")
    rep("   product: y_j = x_2j * x_2j+1.   control: y_j = sin(x_2j) + cos(x_2j+1).")
    rep("   Test MSE / Var(y) after 3000 Adam steps (0 is perfect, 1 is predicting the mean). Three seeds.")
    rows = []
    for kind in ("product", "control"):
        for name in ("GELU, 2 matrices", "SwiGLU, 3 matrices"):
            errs = []
            for seed in range(3):
                torch.manual_seed(seed)
                model = GeluFFN(D, width_gelu) if name.startswith("GELU") else SwiGLU(D, width_glu)
                errs.append(fit(model, torch.nn.Linear(D, PAIRS), kind, seed))
            rows.append([kind, name, " ".join(f"{e:.4f}" for e in errs), f"{sum(errs) / 3:.4f}"])
    rep.table(["target", "FFN", "per seed", "mean"], rows)
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    main()
