"""Section "The KL-regularised objective": the closed-form optimum, checked against brute-force optimisation, and
what beta does to it.

    python labs/teaching-a-language-model-to-follow-instructions/scripts/02_kl_tilt.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prefopt.common import Report  # noqa: E402
from prefopt.tilt import kl, log_partition, objective, optimal_logp, solve_by_gradient  # noqa: E402


def main() -> None:
    rep = Report("02_kl_tilt")
    gen = torch.Generator().manual_seed(0)

    rep.print("1. Closed form vs gradient ascent. 50 completions, random reference and reward.")
    n = 50
    ref = torch.log_softmax(torch.randn(n, generator=gen, dtype=torch.float64) * 2, -1)
    r = torch.randn(n, generator=gen, dtype=torch.float64)
    rows = []
    for beta in [2.0, 0.5, 0.1]:
        exact = optimal_logp(ref, r, beta)
        found = solve_by_gradient(ref, r, beta, steps=4000)
        tv = 0.5 * (exact.exp() - found.exp()).abs().sum().item()
        rows.append({"beta": beta, "J(closed form)": objective(exact, ref, r, beta).item(),
                     "beta*logZ": beta * log_partition(ref, r, beta).item(),
                     "J(gradient ascent)": objective(found, ref, r, beta).item(), "TV(closed, found)": tv})
    rep.table(rows, "{:.6f}")
    rep.print("J at the closed form equals beta log Z, and no optimiser run beats it.")
    rep.data["closed_form"] = rows

    rep.print("\n2. Same support. Five completions; the best one has reference probability exactly 0.")
    ref5 = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0], dtype=torch.float64))
    r5 = torch.tensor([0.0, 0.5, 1.0, 1.5, 10.0], dtype=torch.float64)
    sup = []
    for beta in [10.0, 1.0, 0.1, 0.01]:
        p = optimal_logp(ref5, r5, beta).exp()
        sup.append({"beta": beta, **{f"p(y{i})": p[i].item() for i in range(5)}, "KL": kl(optimal_logp(ref5, r5, beta), ref5).item()})
    rep.table(sup)
    rep.print("y4 has reward 10 and stays at probability 0 for every beta. As beta -> 0, mass pools on y3,")
    rep.print("the best completion the reference can emit.")
    rep.data["support"] = sup

    rep.print("\n3. Exponentially small is nearly unreachable. Same five, with ref(y4) = 1e-12.")
    ref6 = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1 - 1e-12, 1e-12], dtype=torch.float64))
    tiny = []
    for beta in [10.0, 1.0, 0.5, 0.35, 0.3, 0.1]:
        p = optimal_logp(ref6, r5, beta).exp()
        tiny.append({"beta": beta, "p(y4)": p[4].item()})
    rep.table(tiny, "{:.3e}")
    # p(y4) = 1/2 when log(1e-12) + 10/beta = log(0.1) + 1.5/beta (y3 dominates the rest at small beta)
    crossover = 8.5 / (math.log(0.1) - math.log(1e-12))
    rep.print(f"p(y4) passes p(y3) only once beta < {crossover:.3f}: the reward gap 8.5 must beat a log-prior gap of 25.3.")
    rep.data["tiny"] = {"rows": tiny, "crossover_beta": crossover}

    rep.print("\n4. Beta as temperature, on the 50-completion example.")
    sweep = []
    for beta in [100.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01]:
        lp = optimal_logp(ref, r, beta)
        p = lp.exp()
        sweep.append({"beta": beta, "E[r]": (p * r).sum().item(), "KL": kl(lp, ref).item(),
                      "entropy": -(p * lp).sum().item(), "max p": p.max().item()})
    rep.table(sweep)
    rep.print(f"reference: E[r] = {(ref.exp() * r).sum().item():.4f}, best reward = {r.max().item():.4f}, "
              f"entropy = {-(ref.exp() * ref).sum().item():.4f}")
    rep.data["sweep"] = sweep
    rep.save()


if __name__ == "__main__":
    main()
