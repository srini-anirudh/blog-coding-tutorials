"""Two ways to choose mixture weights from a handful of small runs.

RegMix (Liu et al., 2024): fit a regression from weights w to the target loss, then search the
simplex on the fitted surrogate. We use ridge regression on quadratic features instead of
gradient-boosted trees, because it needs only numpy.

Data mixing law (Ye et al., 2024): for each validation domain i, L_i(w) = c_i + k_i * exp(t_i . w).
Fit the parameters on the same runs, then minimise the predicted target.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch


def sample_simplex(n: int, k: int, seed: int = 0) -> np.ndarray:
    """n points drawn uniformly from the k-simplex (Dirichlet with all alphas 1)."""
    return np.random.default_rng(seed).dirichlet(np.ones(k), size=n)


def simplex_grid(k: int, step: float = 0.02) -> np.ndarray:
    m = int(round(1 / step))
    pts = [c for c in itertools.product(range(m + 1), repeat=k - 1) if sum(c) <= m]
    return np.array([[*c, m - sum(c)] for c in pts], dtype=float) / m


def quad_features(w: np.ndarray) -> np.ndarray:
    k = w.shape[1]
    cross = [w[:, i] * w[:, j] for i in range(k) for j in range(i, k)]
    return np.column_stack([np.ones(len(w)), w, *cross])


class QuadraticSurrogate:
    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge

    def fit(self, w: np.ndarray, y: np.ndarray) -> "QuadraticSurrogate":
        X = quad_features(w)
        self.coef = np.linalg.solve(X.T @ X + self.ridge * np.eye(X.shape[1]), X.T @ y)
        return self

    def predict(self, w: np.ndarray) -> np.ndarray:
        return quad_features(w) @ self.coef


class MixingLaw:
    """L_i(w) = c_i + k_i * exp(t_i . w), one law per validation domain, fitted by L-BFGS."""

    def fit(self, w: np.ndarray, L: np.ndarray, steps: int = 300, seed: int = 0) -> "MixingLaw":
        torch.manual_seed(seed)
        W = torch.tensor(w, dtype=torch.float64)
        Y = torch.tensor(L, dtype=torch.float64)  # (n_runs, n_domains)
        n_dom = Y.shape[1]
        lo = Y.min(0).values
        self.log_gap = torch.zeros(n_dom, dtype=torch.float64, requires_grad=True)  # c = min - softplus
        self.log_k = torch.log(Y.mean(0) - lo + 0.1).clone().requires_grad_(True)
        self.t = torch.zeros(n_dom, W.shape[1], dtype=torch.float64, requires_grad=True)
        self.lo = lo
        opt = torch.optim.LBFGS([self.log_gap, self.log_k, self.t], max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = ((self._pred(W) - Y) ** 2).mean()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def _pred(self, W: torch.Tensor) -> torch.Tensor:
        c = self.lo - torch.nn.functional.softplus(self.log_gap)
        return c + self.log_k.exp() * torch.exp(W @ self.t.T)

    def predict(self, w: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._pred(torch.tensor(w, dtype=torch.float64)).numpy()


def leave_one_out_mae(make, w: np.ndarray, y: np.ndarray) -> float:
    """Mean absolute error when each run is predicted from a model fitted on the others."""
    errs = []
    for i in range(len(w)):
        m = np.arange(len(w)) != i
        errs.append(abs(float(np.ravel(make().fit(w[m], y[m]).predict(w[i : i + 1]))[0]) - float(np.ravel(y[i])[0])))
    return float(np.mean(errs))
