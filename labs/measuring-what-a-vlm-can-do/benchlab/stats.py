"""Noise floors, intervals, the blind-test quantities and arena ratings.

Post sections: "The benchmark is too small to separate models", "The question can be answered
without the image" (MMStar's MG and ML) and "Elo: scores from pairwise human votes".
"""

from __future__ import annotations

import math

import torch


def standard_error(p: float, n: int) -> float:
    """SE of an accuracy p measured on n independent items: sqrt(p (1 - p) / n)."""
    return math.sqrt(p * (1 - p) / n)


def bootstrap_ci(correct: torch.Tensor, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of a 0/1 vector."""
    g = torch.Generator().manual_seed(seed)
    x = correct.float()
    idx = torch.randint(0, len(x), (n_boot, len(x)), generator=g)
    means = x[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([alpha / 2, 1 - alpha / 2])).tolist()
    return lo, hi


def paired_bootstrap(a: torch.Tensor, b: torch.Tensor, n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict[str, float]:
    """Interval for mean(a) - mean(b) when both models answered the same items.

    Resampling items (not models) keeps the pairing, so the interval reflects how much the
    difference depends on which items happened to be in the benchmark.
    """
    g = torch.Generator().manual_seed(seed)
    d = a.float() - b.float()
    idx = torch.randint(0, len(d), (n_boot, len(d)), generator=g)
    diffs = d[idx].mean(1)
    lo, hi = torch.quantile(diffs, torch.tensor([alpha / 2, 1 - alpha / 2])).tolist()
    return {"diff": d.mean().item(), "lo": lo, "hi": hi,
            "p_le_0": (diffs <= 0).float().mean().item()}


def multimodal_gain_leakage(s_v: float, s_wv: float, s_t: float) -> dict[str, float]:
    """MMStar's two numbers. MG = S_v - S_wv. ML = max(0, S_wv - S_t)."""
    return {"MG": s_v - s_wv, "ML": max(0.0, s_wv - s_t)}


# --- Arena ratings ---------------------------------------------------------------------------

def elo_win_prob(r_i: float, r_j: float) -> float:
    """P(i beats j) = 1 / (1 + 10^((R_j - R_i) / 400))."""
    return 1.0 / (1.0 + 10 ** ((r_j - r_i) / 400))


def fit_bradley_terry(winners: torch.Tensor, losers: torch.Tensor, n_models: int,
                      steps: int = 500, lr: float = 0.1) -> torch.Tensor:
    """Maximum-likelihood ratings from pairwise votes, on the Elo scale, anchored at mean 1000.

    log P(i beats j) = log sigmoid((R_i - R_j) * ln(10) / 400). Only differences are identified,
    so we fix the mean. A tiny L2 term keeps a model that never lost from running off to infinity.
    """
    # s is the rating in natural log-odds units: P(i beats j) = sigmoid(s_i - s_j).
    s = torch.zeros(n_models, requires_grad=True)
    opt = torch.optim.LBFGS([s], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -torch.nn.functional.logsigmoid(s[winners] - s[losers]).mean() + 1e-6 * (s ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        r = s * 400 / math.log(10)  # 10^(dR/400) = e^(ds)
        return r - r.mean() + 1000
