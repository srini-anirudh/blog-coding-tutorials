"""Section 12, "Which memory comes back": the weighted retrieval score is a log-odds.

Claim: if the features are independent given "useful", the log posterior odds is a constant
plus one log-likelihood ratio per feature. For equal-variance Gaussian features each ratio is
linear, so S = alpha f1 + beta f2 + ... is exactly the naive-Bayes log-odds, and its weights
can be fitted from labels. Because S is a log-odds, it can be thresholded, so retrieval can
return nothing. This module builds a world where that is true and checks each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FEATURES = ("semantic", "recency", "importance")


@dataclass(frozen=True)
class World:
    """Class-conditional Gaussians, one per feature: N(mu1, sd) if useful, N(mu0, sd) if not."""

    mu1: tuple[float, ...] = (0.80, 0.70, 0.65)
    mu0: tuple[float, ...] = (0.30, 0.45, 0.45)
    sd: tuple[float, ...] = (0.15, 0.25, 0.20)
    prior: float = 0.05  # share of (query, memory) pairs that are useful

    def true_weights(self) -> torch.Tensor:
        """d/dx of log N(x; mu1, sd) - log N(x; mu0, sd) = (mu1 - mu0) / sd^2."""
        return torch.tensor([(a - b) / s**2 for a, b, s in zip(self.mu1, self.mu0, self.sd)])

    def true_bias(self) -> float:
        prior_odds = torch.log(torch.tensor(self.prior / (1 - self.prior)))
        quad = sum((b**2 - a**2) / (2 * s**2) for a, b, s in zip(self.mu1, self.mu0, self.sd))
        return float(prior_odds) + quad

    def sample(self, n: int, g: torch.Generator, useful: torch.Tensor | None = None):
        if useful is None:
            useful = torch.rand(n, generator=g) < self.prior
        mu = torch.where(useful[:, None], torch.tensor(self.mu1), torch.tensor(self.mu0))
        x = mu + torch.tensor(self.sd) * torch.randn(n, len(self.sd), generator=g)
        return x, useful


def add_copy_feature(x: torch.Tensor, g: torch.Generator, noise: float = 0.03) -> torch.Tensor:
    """Append a near-copy of recency ("time since last access"): the two are not independent."""
    return torch.cat([x, x[:, 1:2] + noise * torch.randn(len(x), 1, generator=g)], dim=1)


def fit_logistic(x: torch.Tensor, y: torch.Tensor, steps: int = 500) -> tuple[torch.Tensor, float]:
    """Fit log-odds = w.x + b by maximum likelihood (LBFGS on the logistic loss)."""
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    yf = y.float()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, yf)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), float(b.detach())


def naive_bayes_weights(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Gaussian naive Bayes: estimate each feature's ratio on its own, then add them up."""
    x1, x0 = x[y], x[~y]
    mu1, mu0 = x1.mean(0), x0.mean(0)
    var = 0.5 * (x1.var(0) + x0.var(0))
    w = (mu1 - mu0) / var
    prior = y.float().mean()
    b = torch.log(prior / (1 - prior)) + ((mu0**2 - mu1**2) / (2 * var)).sum()
    return w, float(b)


def calibration_error(logits: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    """Expected calibration error of sigmoid(logits) against labels, equal-width bins."""
    p = torch.sigmoid(logits)
    edges = torch.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any():
            err += float(m.float().mean()) * abs(float(p[m].mean()) - float(y[m].float().mean()))
    return err


def confident_bin(logits: torch.Tensor, y: torch.Tensor, cut: float = 0.5) -> tuple[float, float, int]:
    """Among pairs scored above probability `cut`: mean predicted probability, observed rate, count."""
    p = torch.sigmoid(logits)
    m = p > cut
    return float(p[m].mean()), float(y[m].float().mean()), int(m.sum())


def retrieval_episodes(world: World, w: torch.Tensor, b: float, queries: int, pool: int, k: int,
                       threshold: float, p_unanswerable: float, g: torch.Generator) -> dict:
    """Compare top-k against a log-odds threshold on queries with and without a useful memory.

    An answerable query has exactly one useful memory in its pool. An unanswerable one has none.
    """
    stats = {"topk_nonempty_on_unanswerable": 0, "thresh_empty_on_unanswerable": 0, "unanswerable": 0,
             "topk_recall": 0, "thresh_recall": 0, "answerable": 0, "topk_returned": 0, "thresh_returned": 0}
    for _ in range(queries):
        answerable = bool(torch.rand(1, generator=g) >= p_unanswerable)
        useful = torch.zeros(pool, dtype=torch.bool)
        if answerable:
            useful[0] = True
        x, _ = world.sample(pool, g, useful)
        s = x @ w + b
        top = set(torch.topk(s, k).indices.tolist())
        kept = set(torch.nonzero(s > threshold).flatten().tolist())
        stats["topk_returned"] += len(top)
        stats["thresh_returned"] += len(kept)
        if answerable:
            stats["answerable"] += 1
            stats["topk_recall"] += 0 in top
            stats["thresh_recall"] += 0 in kept
        else:
            stats["unanswerable"] += 1
            stats["topk_nonempty_on_unanswerable"] += len(top) > 0
            stats["thresh_empty_on_unanswerable"] += len(kept) == 0
    return stats


# Figure 11's eight memories and its scoring rule, copied from the post's figures3.js.
FIG11_MEMORIES = [
    ("project-x deploys via GitHub Actions", 0.92, 0.95, 0.90),
    ("project-x deploys via Jenkins [2025]", 0.91, 0.10, 0.80),
    ("workflow file: .github/deploy-staging.yml", 0.88, 0.90, 0.85),
    ("deployments go through GitHub Actions", 0.90, 0.93, 0.86),
    ("staging requires OAuth scope read:user", 0.62, 0.88, 0.96),
    ("production deploys need a second approval", 0.58, 0.66, 0.55),
    ("project-y deploys via Cloud Build", 0.55, 0.80, 0.30),
    ("the user prefers long-form writing", 0.05, 0.70, 0.40),
]
FIG11_DUPLICATES = {(0, 3): 0.96}  # the only near-duplicate pair in the figure


def fig11_rank(a: float, b: float, g: float, mmr: bool, penalty: float = 0.45) -> list[tuple[int, float]]:
    """The figure's rule: normalised weighted sum, then subtract 0.45 x similarity from a later near-duplicate."""
    tot = max(a + b + g, 1e-4)
    s = {i: (a * sem + b * rec + g * imp) / tot for i, (_, sem, rec, imp) in enumerate(FIG11_MEMORIES)}
    order = sorted(s, key=s.get, reverse=True)
    if mmr:
        seen: list[int] = []
        for i in order:
            for j in seen:
                sim = FIG11_DUPLICATES.get((min(i, j), max(i, j)), 0.0)
                s[i] -= penalty * sim
            seen.append(i)
        order = sorted(s, key=s.get, reverse=True)
    return [(i, s[i]) for i in order]
