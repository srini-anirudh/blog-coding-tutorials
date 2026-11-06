"""Benign overfitting: where the minimum-norm interpolant puts the noise it was forced to absorb.

For p > n the minimum-norm interpolant is beta_hat = X' (X X')^-1 y. Substituting y = X beta* + eps splits it
into two pieces that we compute separately:

    beta_hat = X' (X X')^-1 X beta*   +   X' (X X')^-1 eps
               (signal, projected)        (noise it absorbed)

The excess risk of each piece is (piece - target)' Sigma (piece - target). The noise piece's risk has the
closed form sigma^2 * tr(G^-1 X Sigma X' G^-1) with G = X X', which is what `noise_risk` returns; no
noise needs to be sampled for it.
"""

import numpy as np

from .linear import GaussianLinearModel


def spiked_model(P, head=6, tail_var=0.05, sigma=0.4):
    """The post's Figure 7: six signal directions with variance 1, then a flat tail of variance 0.05."""
    spec = np.full(P, tail_var)
    spec[:head] = 1.0
    beta = np.zeros(P)
    beta[:head] = [1, -0.8, 0.6, -0.45, 0.35, -0.25]
    return GaussianLinearModel(spec, beta / np.sqrt(np.sum(spec * beta**2)), sigma)


def decompose(model, X, eps, head=6):
    """Split the min-norm interpolant into its signal and noise pieces and measure each.

    Returns the training residual (it should be ~0), the risk of each piece, the total risk, and how much of
    the noise piece's squared norm and risk landed in the head versus the tail directions.
    """
    G = X @ X.T
    y = X @ model.beta_star + eps
    alpha_sig = np.linalg.solve(G, X @ model.beta_star)
    alpha_eps = np.linalg.solve(G, eps)
    b_sig, b_eps = X.T @ alpha_sig, X.T @ alpha_eps
    beta = b_sig + b_eps
    s = model.spec
    noise_risk_dirs = s * b_eps**2
    return {
        "train_residual": float(np.max(np.abs(X @ beta - y))),
        "risk_total": model.risk(beta),
        "risk_signal_piece": model.risk(b_sig),
        "risk_noise_piece": float(noise_risk_dirs.sum()),
        "noise_norm2_tail_share": float(np.sum(b_eps[head:] ** 2) / np.sum(b_eps**2)),
        "noise_risk_tail_share": float(noise_risk_dirs[head:].sum() / noise_risk_dirs.sum()),
        "largest_tail_coef": float(np.max(np.abs(beta[head:]))) if beta.size > head else 0.0,
        "noise_level": model.sigma**2,
    }


def growing_tail(ps, n=30, trials=40, seed=0, head=6):
    """Figure 7 averaged over trials: same n, more and more tail directions available to the model."""
    rows = []
    for p in ps:
        model = spiked_model(p, head=head)
        acc = []
        for t in range(trials):
            rng = np.random.default_rng([seed, t])
            X, _ = model.sample(n, rng)
            acc.append(decompose(model, X, model.sigma * rng.standard_normal(n), head=head))
        rows.append({"p": p, **{k: float(np.median([a[k] for a in acc])) for k in acc[0]}})
    return rows


def noise_risk(X, spec, sigma):
    """Expected excess risk contributed by label noise for the min-norm interpolant, given X."""
    Ginv_X = np.linalg.solve(X @ X.T, X)          # G^-1 X, shape n x p
    return float(sigma**2 * np.sum(Ginv_X**2 * spec))


SPECTRA = {
    # name: function of (k = 1..p, n) -> eigenvalues. Bartlett et al. 2020 classify each as below.
    "isotropic, p = 2n": lambda k, n: np.ones(2 * n),
    "power law k^-2": lambda k, n: k ** -2.0,
    "k^-1 log^-2(k+1)": lambda k, n: 1.0 / (k * np.log(k + 1) ** 2),
    "spike + flat tail of n^2 dims": lambda k, n: np.r_[np.ones(1), np.full(n * n, 1.0 / n)],
}


def spectrum_sweep(ns, p=20000, sigma=1.0, trials=5, seed=0):
    """Noise part of the min-norm interpolant's risk, as n grows, under each spectrum in SPECTRA.

    Benign means this goes to 0 with n: the noise is hit exactly on the training points and still costs
    nothing at test time. Harmful means it stays bounded away from 0."""
    k = np.arange(1, p + 1, dtype=float)
    out = {}
    for name, fn in SPECTRA.items():
        rows = []
        for n in ns:
            spec = fn(k, n)
            vals = []
            for t in range(trials):
                rng = np.random.default_rng([seed, t, n])
                X = rng.standard_normal((n, spec.size)) * np.sqrt(spec)
                vals.append(noise_risk(X, spec, sigma))
            rows.append({"n": n, "p": int(spec.size), "noise_risk_mean": float(np.mean(vals))})
        out[name] = rows
    return out
