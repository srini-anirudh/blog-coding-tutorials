"""Section "The offline preference family": seven opinions about one margin axis (the post's Figure 9).

Each loss is evaluated with autograd at the same margins h. The losses look alike; the gradients do not.

    python labs/teaching-a-language-model-to-follow-instructions/scripts/04_loss_family.py --beta 1.0 --target 1.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prefopt.common import Report  # noqa: E402
from prefopt.losses import FAMILY, loss_and_grad  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--target", type=float, default=1.6, help="target margin: SLiC's delta and SimPO's gamma/beta")
    args = ap.parse_args()
    rep = Report("04_loss_family")
    h = torch.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    rep.print(f"beta = {args.beta}, target margin = {args.target}. h is the preference margin.")
    out = {}
    for what in ("loss", "dloss/dh"):
        rep.print(f"\n{what}")
        rows = []
        for name in FAMILY:
            loss, grad = loss_and_grad(name, h, args.beta, args.target)
            vals = loss if what == "loss" else grad
            rows.append({"method": name, **{f"h={x:g}": v.item() for x, v in zip(h.tolist(), vals)}})
            out.setdefault(name, {})[what] = vals.tolist()
        rep.table(rows, "{:.3f}")

    _, g_dpo = loss_and_grad("DPO", h, args.beta, args.target)
    closed = -args.beta * torch.sigmoid(-args.beta * h)
    rep.print(f"\nDPO gradient vs -beta * sigmoid(-beta h): max gap {(g_dpo - closed).abs().max().item():.1e}")
    rep.print("What to read off the gradient table:")
    rep.print("  DPO and SimPO fade smoothly as h grows. SimPO is DPO shifted right by its target margin.")
    rep.print("  ORPO's odds term has DPO's shape; it differs in what h measures (log-odds, no reference).")
    rep.print("  KTO pushes hardest at its reference point and fades on both sides, so badly wrong examples")
    rep.print("  get little push too.")
    rep.print("  The two hinges give a constant push and then stop dead: exactly zero past their margin.")
    rep.print(f"  IPO never stops. Past h = 1/(2 beta) = {1 / (2 * args.beta):g} its gradient changes sign and pulls")
    rep.print("  the margin back. It is the only member that refuses to grow the margin without bound.")
    rep.data.update({"beta": args.beta, "target": args.target, "h": h.tolist(), "family": out})
    rep.save()


if __name__ == "__main__":
    main()
