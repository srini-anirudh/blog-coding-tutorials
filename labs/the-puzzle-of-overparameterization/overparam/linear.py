"""Least squares, solved exactly: the bias-variance U, the interpolation threshold and double descent.

One solver does all the work. `spectral_fit` writes ridge, minimum-norm least squares and t steps of
gradient descent as three filters on the singular values s_i of the design matrix. The coefficient on
direction i is (u_i . y) / s_i, so a tiny s_i blows the fit up unless something damps it. That is the
post's conditioning story in one line.

The double-descent model is the one the post's Figures 4 to 6 and Lab 1 compute in the browser:
x ~ N(0, Sigma) with Sigma = diag(1, 1/2, 1/3, ...), y = <beta*, x> + sigma * noise, and a model that
may only use the first p coordinates. Test error is the exact excess risk (beta - beta*)' Sigma (beta - beta*).
"""

from dataclasses import dataclass

import numpy as np


def spectral_fit(X, y, ridge=0.0, steps=None, lr=None):
    """Fit y ~ X @ beta by filtering the singular values of X.

    ridge > 0             : ridge regression, filter s / (s^2 + ridge).
    ridge = 0, steps None : minimum-norm least squares, filter 1 / s (the pseudoinverse).
    steps = t             : t steps of gradient descent on 0.5 ||X beta - y||^2 from beta = 0 with step
                            size lr (default 1 / s_max^2), filter (1 - (1 - lr s^2)^t) / s.
                            Directions with large s are fitted first, tiny ones last.
    """
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    keep = s > s.max() * 1e-12
    u, s, vt = u[:, keep], s[keep], vt[keep]
    if steps is not None:
        lr = lr if lr is not None else 1.0 / s.max() ** 2
        with np.errstate(divide="ignore"):  # lr s^2 = 1 gives log1p(-1) = -inf, which is the right limit
            filt = -np.expm1(steps * np.log1p(-lr * s**2)) / s
    else:
        filt = s / (s**2 + ridge)
    return vt.T @ (filt * (u.T @ y))


# ---------------------------------------------------------------------------- Figure 1: bias and variance

def fig1_target(x):
    return np.sin(2.3 * x) + 0.42 * x * x


def bias_variance(degrees, n=14, sigma=0.24, redraws=20, seed=0):
    """The post's Figure 1 setting: n equispaced inputs on [-1, 1], fresh noise for every redraw,
    one polynomial fit per degree. Returns bias^2, variance and noise averaged over a test grid.
    The three sum to the expected squared error on a fresh noisy label."""
    rng = np.random.default_rng(seed)
    x = -1 + 2 * (np.arange(n) + 0.5) / n
    grid = np.linspace(-1, 1, 400)
    preds = {d: np.empty((redraws, grid.size)) for d in degrees}
    for r in range(redraws):
        y = fig1_target(x) + sigma * rng.standard_normal(n)
        for d in degrees:
            # Legendre basis: same function class as plain powers, better conditioned.
            beta = spectral_fit(np.polynomial.legendre.legvander(x, d), y)
            preds[d][r] = np.polynomial.legendre.legvander(grid, d) @ beta
    rows = []
    for d in degrees:
        bias2 = float(np.mean((preds[d].mean(0) - fig1_target(grid)) ** 2))
        var = float(np.mean(preds[d].var(0)))
        rows.append({"degree": d, "bias2": bias2, "variance": var, "noise": sigma**2,
                     "total": bias2 + var + sigma**2})
    return rows


# ---------------------------------------------------------------------------- Figures 4-6: double descent

@dataclass
class GaussianLinearModel:
    """x ~ N(0, diag(spec)), y = <beta_star, x> + sigma * noise."""
    spec: np.ndarray
    beta_star: np.ndarray
    sigma: float

    def sample(self, n, rng):
        X = rng.standard_normal((n, self.spec.size)) * np.sqrt(self.spec)
        return X, X @ self.beta_star + self.sigma * rng.standard_normal(n)

    def risk(self, beta):
        """Exact excess test risk of a model that uses the first len(beta) coordinates."""
        full = np.zeros_like(self.beta_star)
        full[: beta.size] = beta
        return float(np.sum(self.spec * (full - self.beta_star) ** 2))


def harmonic_model(P=420, sigma=0.45):
    """Sigma = diag(1/j) and beta*_j proportional to j^-0.6, normalized to unit signal variance."""
    j = np.arange(1, P + 1)
    spec = 1.0 / j
    beta = j ** -0.6
    return GaussianLinearModel(spec, beta / np.sqrt(np.sum(spec * beta**2)), sigma)


def model_wise(ps, n=40, sigma=0.45, ridge=0.0, trials=40, seed=0):
    """Test risk, coefficient norm and smallest singular value as the model uses p = 1..P coordinates."""
    model = harmonic_model(sigma=sigma)
    rows = []
    for p in ps:
        risk, norm, smin = [], [], []
        for t in range(trials):
            X, y = model.sample(n, np.random.default_rng([seed, t]))
            beta = spectral_fit(X[:, :p], y, ridge=ridge)
            risk.append(model.risk(beta))
            norm.append(np.linalg.norm(beta))
            smin.append(np.linalg.svd(X[:, :p], compute_uv=False).min())
        rows.append({"p": p, "risk_median": float(np.median(risk)), "risk_mean": float(np.mean(risk)),
                     "norm_median": float(np.median(norm)), "smallest_sv_median": float(np.median(smin))})
    return rows


def tuned_ridge(ps, n=40, sigma=0.45, ridges=np.logspace(-6, 2, 33), trials=40, seed=0):
    """For each p, the ridge in `ridges` with the lowest mean risk (an oracle choice, as in the
    Nakkiran-Venkat-Kakade-Ma experiments), and that risk."""
    model = harmonic_model(sigma=sigma)
    rows = []
    for p in ps:
        risks = np.zeros((trials, len(ridges)))
        for t in range(trials):
            X, y = model.sample(n, np.random.default_rng([seed, t]))
            for k, lam in enumerate(ridges):
                risks[t, k] = model.risk(spectral_fit(X[:, :p], y, ridge=lam))
        best = int(np.argmin(risks.mean(0)))
        rows.append({"p": p, "best_ridge": float(ridges[best]), "risk_mean": float(risks.mean(0)[best]),
                     "risk_median": float(np.median(risks[:, best]))})
    return rows


def sample_wise(ns, p=60, sigma=0.45, trials=40, seed=0):
    """Fixed p, growing n. Adding data moves the threshold n = p onto the model."""
    model = harmonic_model(sigma=sigma)
    rows = []
    for n in ns:
        risk = [model.risk(spectral_fit(*_first_p(model.sample(n, np.random.default_rng([seed, t, 1])), p)))
                for t in range(trials)]
        rows.append({"n": n, "risk_median": float(np.median(risk)), "risk_mean": float(np.mean(risk))})
    return rows


def _first_p(sample, p):
    X, y = sample
    return X[:, :p], y


def epoch_wise(steps, n=40, p=44, sigma=0.45, trials=40, seed=0):
    """One fixed model (the post uses n = 40, p = 44), more gradient-descent steps. Returns median test risk
    and training MSE at each step count, plus the min-norm end point gradient descent converges to."""
    model = harmonic_model(sigma=sigma)
    risk = np.zeros((trials, len(steps)))
    train = np.zeros((trials, len(steps)))
    end = []
    for t in range(trials):
        X, y = model.sample(n, np.random.default_rng([seed, t, 2]))
        X = X[:, :p]
        for j, k in enumerate(steps):
            beta = spectral_fit(X, y, steps=k)
            risk[t, j] = model.risk(beta)
            train[t, j] = np.mean((X @ beta - y) ** 2)
        end.append(model.risk(spectral_fit(X, y)))
    rows = [{"steps": int(k), "risk_median": float(np.median(risk[:, j])),
             "train_mse_median": float(np.median(train[:, j]))} for j, k in enumerate(steps)]
    return rows, float(np.median(end))
