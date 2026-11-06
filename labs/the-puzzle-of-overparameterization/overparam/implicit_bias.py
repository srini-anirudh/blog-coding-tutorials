"""The algorithm picks the solution: gradient descent on separable logistic regression.

Soudry et al. (2018): with no regularizer, w(t) / ||w(t)|| converges to the hard-margin SVM direction,
||w(t)|| grows like log t, and the angle to the SVM direction shrinks like 1 / log t. We compute the SVM
exactly (dual coordinate ascent) and watch gradient descent walk towards it.

Gunasekar et al. (2018): steepest descent in another norm heads for the max-margin solution in *that* norm.
`steepest_descent` runs the l2 (gradient descent), l1 (greedy coordinate descent) and l-infinity
(sign descent) versions on the same data so their answers can be compared.
"""

import numpy as np


def separable_data(n, d=2, gap=0.1, seed=0, w_true=None):
    """Gaussian inputs with an anisotropic covariance, labelled by a separator through the origin.
    Points closer than `gap` to the separator are dropped, so the training set is separable."""
    rng = np.random.default_rng(seed)
    if w_true is None:
        w_true = np.r_[1.0, 0.5, np.zeros(d - 2)] if d >= 2 else np.ones(1)
    w_true = w_true / np.linalg.norm(w_true)
    scale = np.linspace(1.5, 0.5, d)
    X, y = [], []
    while len(X) < n:
        x = rng.standard_normal(d) * scale
        m = x @ w_true
        if abs(m) >= gap:
            X.append(x)
            y.append(np.sign(m))
    return np.array(X), np.array(y), w_true


def hard_margin_svm(X, y, sweeps=20000, tol=1e-12):
    """Hard-margin SVM without a bias term, by coordinate ascent on the dual
    max sum(a) - 0.5 ||sum a_i y_i x_i||^2 subject to a >= 0. Returns w with min_i y_i w.x_i = 1."""
    Z = X * y[:, None]
    a = np.zeros(len(y))
    w = np.zeros(X.shape[1])
    sq = np.sum(Z**2, axis=1)
    for _ in range(sweeps):
        biggest = 0.0
        for i in range(len(y)):
            new = max(0.0, a[i] + (1.0 - Z[i] @ w) / sq[i])
            if new != a[i]:
                w += (new - a[i]) * Z[i]
                biggest = max(biggest, abs(new - a[i]))
                a[i] = new
        if biggest < tol:
            break
    return w


def angle(u, v):
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def logistic_loss_grad(w, Z):
    """Mean logistic loss over margins Z @ w (Z has rows y_i x_i) and its gradient, computed stably."""
    m = Z @ w
    loss = np.mean(np.logaddexp(0.0, -m))
    sig = np.exp(-np.logaddexp(0.0, m))           # sigmoid(-m)
    return float(loss), -(Z * sig[:, None]).mean(0)


def gd_logistic(X, y, checkpoints, lr=1.0, weight_decay=0.0):
    """Plain gradient descent from w = 0 on the mean logistic loss (+ weight_decay/2 ||w||^2).
    Returns a dict {step: w} at the requested steps."""
    Z = X * y[:, None]
    w = np.zeros(X.shape[1])
    want, out = set(int(c) for c in checkpoints), {}
    for t in range(1, max(want) + 1):
        _, g = logistic_loss_grad(w, Z)
        w = w - lr * (g + weight_decay * w)
        if t in want:
            out[t] = w.copy()
    return out


def margin(w, X, y, norm=2):
    """Smallest normalized margin min_i y_i w.x_i / ||w||_norm (norm = 1, 2 or np.inf)."""
    return float(np.min(y * (X @ w)) / np.linalg.norm(w, ord=norm))


def steepest_descent(X, y, geometry, steps=20000, lr=0.05):
    """Steepest descent on the logistic loss with respect to a norm, with a normalized step.

    l2   : step along -g / ||g||_2                  (normalized gradient descent)
    l1   : step along the single coordinate with the largest |g_j|  (greedy coordinate descent)
    linf : step along -sign(g)                      (sign descent)
    """
    Z = X * y[:, None]
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        # Only the direction of the gradient matters for a normalized step. Each point's weight is
        # sigmoid(-m_i); dividing all weights by the largest one, in log space, avoids underflow.
        log_wt = -np.logaddexp(0.0, Z @ w)
        g = -(Z * np.exp(log_wt - log_wt.max())[:, None]).sum(0)
        if geometry == "l2":
            d = g / np.linalg.norm(g)
        elif geometry == "l1":
            d = np.zeros_like(g)
            j = int(np.argmax(np.abs(g)))
            d[j] = np.sign(g[j])
        elif geometry == "linf":
            d = np.sign(g)
        else:
            raise ValueError(geometry)
        w = w - lr * d
    return w
