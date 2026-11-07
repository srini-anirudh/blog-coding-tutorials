"""Section "The offline preference family": seven losses on one margin axis, and the gradients that tell them apart.

Each loss is written as a function of its own margin m (see prefopt/dpo.py for what m means per method). We print
the loss and dL/dm from autograd on a grid. The post's claim is that the methods differ mostly in the gradient:
whether it ever reaches zero, and whether it changes sign.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from prefopt.dpo import FAMILY  # noqa: E402
from prefopt.report import Report  # noqa: E402

GRID = [-3.0, -1.0, 0.0, 0.5, 1.0, 1.6, 3.0, 6.0]


def curves():
    out = {}
    for name, fn in FAMILY.items():
        m = torch.tensor(GRID, dtype=torch.float64, requires_grad=True)
        loss = fn(m)
        (grad,) = torch.autograd.grad(loss.sum(), m)
        out[name] = {"loss": loss.detach().tolist(), "grad": grad.tolist()}
    return out


def describe(grad: list[float]) -> str:
    g = torch.tensor(grad)
    if bool((g > 0).any()) and bool((g < 0).any()):
        return "changes sign at its target: pulls large margins back"
    if float(g[-1]) == 0.0:
        return "exactly zero past a threshold"
    if abs(float(g[0])) < abs(float(g[GRID.index(0.0)])):
        return "vanishes at both ends: very wrong pairs barely move"
    return "never zero, decays toward zero"


def main():
    rep = Report("06_loss_family")
    rep.line("Defaults: beta = 1, IPO tau = 1 (target margin 0.5), SLiC delta = 1, SimPO gamma = 1, KTO z0 = 0")
    c = curves()
    for key in ("loss", "grad"):
        rep.line(f"\n{'Loss' if key == 'loss' else 'dLoss/dm (autograd)'} at margin m")
        rep.table(["method"] + [f"m={x:g}" for x in GRID], [[n] + c[n][key] for n in c])
    rep.line("\nShape of the gradient over the grid")
    rep.table(["method", "gradient"], [[n, describe(c[n]["grad"])] for n in c])
    rep.data = {"grid": GRID, "curves": c}
    rep.save()


if __name__ == "__main__":
    main()
