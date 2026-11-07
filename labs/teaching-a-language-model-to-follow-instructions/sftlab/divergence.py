"""Entropy, cross-entropy and KL for discrete and gridded 1-D distributions."""

from __future__ import annotations

import numpy as np


def entropy(d):
    d = np.asarray(d, float)
    m = d > 0
    return float(-(d[m] * np.log(d[m])).sum())


def cross_entropy(d, p):
    d, p = np.asarray(d, float), np.asarray(p, float)
    m = d > 0
    return float(np.inf) if np.any(p[m] == 0) else float(-(d[m] * np.log(p[m])).sum())


def kl(d, p):
    """KL(d || p) in nats; infinite if p is 0 where d is not."""
    d, p = np.asarray(d, float), np.asarray(p, float)
    m = d > 0
    if np.any(p[m] == 0):
        return float("inf")
    return float((d[m] * np.log(d[m] / p[m])).sum())


def normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def mixture_pdf(x, weights, mus, sds):
    return sum(w * normal_pdf(x, m, s) for w, m, s in zip(weights, mus, sds))


def grid_kl(x, a, b, eps=1e-300):
    """KL(a || b) for densities on a grid."""
    a, b = np.maximum(a, eps), np.maximum(b, eps)
    return float(np.trapezoid(a * np.log(a / b), x))


def fit_gaussian(x, target, direction: str):
    """Best single Gaussian by grid search over (mu, sd). direction "forward" minimises KL(target||q),
    "reverse" minimises KL(q||target)."""
    best = (None, None, np.inf)
    for mu in np.linspace(-6, 6, 121):
        for sd in np.linspace(0.3, 6, 115):
            q = normal_pdf(x, mu, sd)
            v = grid_kl(x, target, q) if direction == "forward" else grid_kl(x, q, target)
            if v < best[2]:
                best = (mu, sd, v)
    return best
