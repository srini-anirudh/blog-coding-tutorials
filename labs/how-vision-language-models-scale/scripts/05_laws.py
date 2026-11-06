"""The scaling-law forms in the post, solved by hand.

    python labs/how-vision-language-models-scale/scripts/05_laws.py

1. Chinchilla allocation: closed form against brute force, with Hoffmann et al.'s published fit.
2. Fitting: recover known constants from noisy synthetic losses (a check of the fitter, not a result).
3. Fusion splits: how far N/D drifts apart for the exponents Shukor et al. (2025) report.
4. Modality interaction (Aghajanyan et al., 2023): competition turns into synergy as N grows.
5. Inference-optimal LLM size against visual tokens (Li et al., 2024).
Every constant not taken from a paper is marked illustrative.
"""

import numpy as np
from _common import Report, table

from vlmscale.laws import (chinchilla_loss, compute_optimal, compute_optimal_numeric, crossover_size,
                           fit_chinchilla, inference_frontier, interaction)

r = Report("laws")

# 1 ---------------------------------------------------------------- Chinchilla
E, A, B, AL, BE = 1.69, 406.4, 410.7, 0.34, 0.28  # Hoffmann et al. (2022), approach 3
r("== 1. Compute-optimal N and D for L = E + A/N^a + B/D^b, C = 6ND (Hoffmann et al. constants) ==")
rows = []
for c in (1e20, 1e21, 1e22, 1e23, 1e24):
    n, d = compute_optimal(c, A, B, AL, BE)
    n2, _ = compute_optimal_numeric(c, E, A, B, AL, BE)
    rows.append([f"{c:.0e}", f"{n / 1e9:.2f}B", f"{n2 / 1e9:.2f}B", f"{d / 1e9:.0f}B", f"{d / n:.1f}"])
r(table(rows, ["C (FLOPs)", "N_opt closed form", "N_opt brute force", "D_opt tokens", "tokens / param"]))
r(f"N_opt grows as C^{BE / (AL + BE):.2f}. With these constants the ratio D/N drifts with C; the 20 tokens per")
r("parameter rule of thumb comes from Hoffmann et al.'s other two approaches.")

# 2 ---------------------------------------------------------------- fitting
r("\n== 2. Fit check: synthetic losses from known constants plus 1% noise (illustrative data) ==")
rng = np.random.default_rng(0)
ns, ds = np.meshgrid(np.logspace(6, 9, 7), np.logspace(8, 11, 7))
ns, ds = ns.ravel(), ds.ravel()
truth = dict(e=1.8, a_coef=300.0, b_coef=900.0, alpha=0.32, beta=0.30)
loss = chinchilla_loss(ns, ds, **truth) * np.exp(rng.normal(0, 0.01, ns.shape))
fit = fit_chinchilla(ns, ds, loss)
r(table([["E", truth["e"], f"{fit['E']:.3f}"], ["A", truth["a_coef"], f"{fit['A']:.1f}"],
         ["B", truth["b_coef"], f"{fit['B']:.1f}"], ["alpha", truth["alpha"], f"{fit['alpha']:.3f}"],
         ["beta", truth["beta"], f"{fit['beta']:.3f}"]], ["constant", "true", "fitted"]))
r("The exponents come back within 0.03. E, A and B are less well pinned, because they trade off against each other.")

# 3 ---------------------------------------------------------------- fusion splits
r("\n== 3. Same loss slope, different split: N/D ~ C^b with Shukor et al. (2025) exponents ==")
r("Normalised so every design has N/D = 1/20 at C = 1e20 (illustrative intercept).")
exps = {"early fusion": 0.053, "late fusion": 0.076, "early fusion MoE": -0.312}
rows = []
for c in (1e20, 1e22, 1e24):
    rows.append([f"{c:.0e}"] + [f"{(1 / 20) * (c / 1e20) ** b:.4f}" for b in exps.values()])
r(table(rows, ["C"] + [f"N/D {k}" for k in exps]))
r("Loss exponents reported: -0.0492 (early), -0.0494 (late), -0.0474 (MoE). Late fusion wants more parameters")
r("per token as C grows; the sparse model shifts its extra budget into data.")

# 4 ---------------------------------------------------------------- interaction
r("\n== 4. Modality interaction term -C + A/N^a + B/D^b (illustrative constants) ==")
ic = dict(c_benefit=0.05, a_coef=3.0, alpha=0.25, b_coef=2.0, beta=0.2)
d = 45e9
rows = [[f"{n:.0e}", f"{float(interaction(n, d, **ic)):+.4f}",
         "competition" if interaction(n, d, **ic) > 0 else "synergy"] for n in (1e7, 1e8, 1e9, 1e10, 3e10, 1e11)]
r(table(rows, ["N", "interaction", "regime"]))
r(f"Crossover at N = {crossover_size(d, **ic):.2e} for D = 45B tokens with these constants.")

# 5 ---------------------------------------------------------------- inference-optimal
r("\n== 5. Fixed inference budget N (Q + V): spend on LLM size or on visual tokens? ==")
r("alpha = 0.077, beta = 0.015 from Li et al. (2024); A = B = 1 and the OCR setting are illustrative.")
budget, q = 300.0, 50  # B-param x tokens; prompt tokens
vs = [1, 4, 16, 36, 64, 144, 256, 576]
for label, b_coef, beta in (("visual reasoning", 1.0, 0.015), ("OCR-like (beta = 0.25, B x 3)", 3.0, 0.25)):
    rows_v = inference_frontier(budget, q, vs, 1.0, b_coef, 0.077, beta)
    best = min((x for x in rows_v if x["available"]), key=lambda x: x["error"])
    r(f"\n{label}: budget {budget:.0f} B-param x tokens, Q = {q}")
    r(table([[x["V"], f"{x['N']:.2f}", "yes" if x["available"] else "no", f"{x['error']:.4f}",
              "<- best" if x is best else ""] for x in rows_v], ["V", "N (B)", "0.5B-72B?", "error", ""]))
r("\nLonger prompts move the reasoning optimum toward more visual tokens (the post's third observation):")
rows = []
for q_len in (0, 10, 50, 200):
    fr = [x for x in inference_frontier(budget, q_len, range(1, 577), 1.0, 1.0, 0.077, 0.015) if x["available"]]
    rows.append([q_len, min(fr, key=lambda x: x["error"])["V"]])
r(table(rows, ["prompt tokens Q", "best V (visual reasoning)"]))
r("\nscripts/06_tokens_vs_size.py trains real small models to test the same trade-off.")
r.save()
