"""Rank from first principles (post: "Rank, from first principles", Fig. 2).

1. Rebuild a 24 x 24 matrix one singular triplet at a time and check that the error equals the spectrum's tail
   (Eckart-Young), for a fast-decaying and a slowly decaying spectrum.
2. Do what LoRA does instead: never compute an SVD, start from B = 0, and search the rank-r set with Adam.
   Compare the error it reaches with the Eckart-Young optimum.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loralab.report import Report, table
from loralab.spectrum import decay_spectrum, eckart_young_error, fit_factored, matrix_with_spectrum, truncate


def main():
    rep = Report("02_rank")
    rep("## Part 1: rank-r truncation of a 24 x 24 matrix (seed 0)")
    ey = {}
    for kind in ("fast", "slow"):
        M = matrix_with_spectrum(decay_spectrum(24, kind), 24, 24)
        rows = []
        for r in (1, 2, 4, 8, 12, 16, 20, 24):
            err = (M - truncate(M, r)).norm().item()
            tail = eckart_young_error(M, r)
            rows.append([r, err, tail, err / M.norm().item()])
        ey[kind] = rows
        rep(f"\n{kind} decay: sigma_1 = {decay_spectrum(24, kind)[0]:.1f}, sigma_24 = {decay_spectrum(24, kind)[-1]:.3g}")
        for line in table(rows, ["rank", "||M - M_r||_F", "sqrt(tail sum sigma^2)", "relative error"]):
            rep(line)
    rep.data["truncation"] = ey

    rep("\n## Part 2: LoRA-style search (B = 0, A random, Adam lr 1e-2, 3000 steps) vs the Eckart-Young optimum")
    rows = []
    for kind, shape in (("fast", (24, 24)), ("slow", (24, 24)), ("power-law", (256, 256))):
        if kind == "power-law":
            sig = 10.0 * torch.arange(1, 257, dtype=torch.float64) ** -1.0
        else:
            sig = decay_spectrum(24, kind)
        M = matrix_with_spectrum(sig, *shape, seed=1)
        for r in (1, 4, 8, 16):
            fit, _ = fit_factored(M, r)
            best = eckart_young_error(M, r)
            rows.append([kind, f"{shape[0]}x{shape[1]}", r, best, fit, fit / best])
    for line in table(rows, ["spectrum", "shape", "rank", "Eckart-Young", "Adam on B A", "ratio"]):
        rep(line)
    rep("A ratio near 1 means gradient descent on the factors found the best rank-r matrix without an SVD.")
    rep("This is a pure approximation problem. In fine-tuning there is no fixed target matrix to approximate.")
    rep.data["factored_fit"] = rows
    rep.save()


if __name__ == "__main__":
    main()
