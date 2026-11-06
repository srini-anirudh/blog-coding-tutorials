"""Figure 11 and section 12: the retrieval score as naive-Bayes log-odds, fitted weights, and thresholding.

    python labs/memory-is-more-than-context/scripts/05_scoring.py
"""

from __future__ import annotations

import torch

from _common import Report

from memlab.scoring import (FEATURES, FIG11_MEMORIES, World, add_copy_feature, calibration_error, confident_bin,
                            fig11_rank, fit_logistic, naive_bayes_weights, retrieval_episodes)

out = Report("05_scoring")
data: dict = {}
torch.manual_seed(0)
g = torch.Generator().manual_seed(0)
world = World()

out("1. Figure 11's eight memories, ranked by the figure's own rule")
for label, (a, b, c, mmr) in {"semantic only": (1, 0, 0, False), "a=.60 b=.25 g=.30": (0.6, 0.25, 0.3, False),
                              "a=.60 b=.25 g=.30 + MMR": (0.6, 0.25, 0.3, True)}.items():
    ranked = fig11_rank(a, b, c, mmr)
    out(f"  {label:<26}" + "  ".join(f"#{i}:{s:.2f}" for i, s in ranked))
    data[f"fig11/{label}"] = [[i, round(s, 4)] for i, s in ranked]
out("  " + "; ".join(f"#{i} {m[0]}" for i, m in enumerate(FIG11_MEMORIES)))

x, y = world.sample(200_000, g)
xt, yt = world.sample(200_000, g)
true_w, true_b = world.true_weights(), world.true_bias()
fit_w, fit_b = fit_logistic(x, y)
nb_w, nb_b = naive_bayes_weights(x, y)
out(f"\n2. Fitting the weights from labels (200,000 labelled pairs, prior P(useful) = {world.prior})")
out(f"  {'':<22}" + "".join(f"{f:>12}" for f in FEATURES) + f"{'bias':>10}")
for name, w, b in (("true log-LR slopes", true_w, true_b), ("logistic regression", fit_w, fit_b),
                   ("naive Bayes", nb_w, nb_b)):
    out(f"  {name:<22}" + "".join(f"{v:>12.2f}" for v in w.tolist()) + f"{b:>10.2f}")
data["weights"] = {"true": true_w.tolist() + [true_b], "logistic": fit_w.tolist() + [fit_b], "nb": nb_w.tolist() + [nb_b]}

out("\n3. Add a fourth feature that nearly copies recency (time since last access)")
x4, xt4 = add_copy_feature(x, g), add_copy_feature(xt, g)
fit4_w, fit4_b = fit_logistic(x4, y)
nb4_w, nb4_b = naive_bayes_weights(x4, y)
out(f"  {'':<22}" + "".join(f"{f:>12}" for f in FEATURES + ('last access',)) + f"{'bias':>10}")
out(f"  {'logistic regression':<22}" + "".join(f"{v:>12.2f}" for v in fit4_w.tolist()) + f"{fit4_b:>10.2f}")
out(f"  {'naive Bayes':<22}" + "".join(f"{v:>12.2f}" for v in nb4_w.tolist()) + f"{nb4_b:>10.2f}")
out(f"  held-out calibration           ECE     pairs scored P>0.5: mean predicted / observed useful")
for name, logits in (("3 features, naive Bayes", xt @ nb_w + nb_b), ("4 features, logistic", xt4 @ fit4_w + fit4_b),
                     ("4 features, naive Bayes", xt4 @ nb4_w + nb4_b)):
    ece = calibration_error(logits, yt)
    mp, obs, n = confident_bin(logits, yt)
    out(f"  {name:<28}{ece:>8.4f}     {mp:.3f} / {obs:.3f}  (n={n})")
    data[f"calibration/{name}"] = {"ece": round(ece, 5), "mean_pred_above_half": round(mp, 4),
                                   "observed_above_half": round(obs, 4), "n": n}

out("\n4. Top-k against a log-odds threshold (fitted 3-feature score, threshold log-odds > 0, i.e. P > 0.5)")
out("  2,000 queries, pools of 50 memories, 30% of queries have no useful memory, k = 3")
st = retrieval_episodes(world, fit_w, fit_b, 2000, 50, 3, 0.0, 0.3, g)
un, an = st["unanswerable"], st["answerable"]
out(f"  {'':<34}{'top-3':>9}{'threshold':>11}")
out(f"  {'unanswerable: returned nothing':<34}{0:>9.3f}{st['thresh_empty_on_unanswerable'] / un:>11.3f}")
out(f"  {'answerable: useful memory returned':<34}{st['topk_recall'] / an:>9.3f}{st['thresh_recall'] / an:>11.3f}")
out(f"  {'memories returned per query':<34}{st['topk_returned'] / 2000:>9.2f}{st['thresh_returned'] / 2000:>11.2f}")
data["threshold"] = st
out.save(data)
