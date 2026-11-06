"""The scaling-law forms the post uses, with the fitting and optimisation done by hand.

- Chinchilla:            L(N, D) = E + A / N^a + B / D^b, compute C = 6 N D
- Inference-optimal:     Y(N, V) = A / N^a + B / V^b + D at fixed cost N (Q + V)      (Li et al., 2024)
- Modality interaction:  the Aghajanyan et al. (2023) term  -C + A / N^a + B / D^b
- Utility decay:         b_{k+1} = b (1/2)^(k / tau), piecewise power laws per epoch  (Goyal et al., 2024)
"""

from __future__ import annotations

import math

import numpy as np
import torch


# ---------------------------------------------------------------- Chinchilla
def chinchilla_loss(n, d, e, a_coef, b_coef, alpha, beta):
    return e + a_coef / np.power(n, alpha) + b_coef / np.power(d, beta)


def compute_optimal(c: float, a_coef: float, b_coef: float, alpha: float, beta: float) -> tuple[float, float]:
    """Minimise L over N with D = C / 6N. Setting dL/dN = 0 gives

        N_opt = G (C/6)^(beta/(alpha+beta)),  G = (alpha A / (beta B))^(1/(alpha+beta)),  D_opt = C / (6 N_opt).
    """
    g = (alpha * a_coef / (beta * b_coef)) ** (1.0 / (alpha + beta))
    n = g * (c / 6.0) ** (beta / (alpha + beta))
    return n, c / (6.0 * n)


def compute_optimal_numeric(c: float, e: float, a_coef: float, b_coef: float, alpha: float, beta: float,
                            grid: int = 20001) -> tuple[float, float]:
    """The same minimum found by brute force on a log grid, as a check on the closed form."""
    log_n = np.linspace(0, math.log(c / 6.0), grid)
    n = np.exp(log_n)
    loss = chinchilla_loss(n, c / (6.0 * n), e, a_coef, b_coef, alpha, beta)
    i = int(np.argmin(loss))
    return float(n[i]), float(c / (6.0 * n[i]))


def fit_chinchilla(n, d, loss, inits: int = 64, steps: int = 300, seed: int = 0) -> dict:
    """Fit (E, A, B, alpha, beta) by minimising a Huber loss on log L, from many random starts.

    This follows Hoffmann et al.'s approach 3: parameterise A = exp(a), B = exp(b), E = exp(e) and
    fit with L-BFGS. Returns the best fit by final objective.
    """
    n_t = torch.tensor(np.asarray(n, dtype=np.float64))
    d_t = torch.tensor(np.asarray(d, dtype=np.float64))
    log_l = torch.log(torch.tensor(np.asarray(loss, dtype=np.float64)))
    gen = torch.Generator().manual_seed(seed)
    best = None
    for _ in range(inits):
        init = torch.tensor([0.0, 0.0, -1.0, 0.0, 0.0], dtype=torch.float64)
        init += torch.rand(5, generator=gen, dtype=torch.float64) * torch.tensor([10.0, 10.0, 2.0, 1.0, 1.0],
                                                                                 dtype=torch.float64)
        theta = init.clone().requires_grad_(True)  # a, b, e, alpha, beta
        opt = torch.optim.LBFGS([theta], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

        def value():
            a, b, e, al, be = theta
            pred = torch.logsumexp(torch.stack([a - al * torch.log(n_t), b - be * torch.log(d_t),
                                                e.expand_as(n_t)]), dim=0)
            return torch.nn.functional.huber_loss(pred, log_l, delta=1e-3, reduction="sum")

        def objective():
            opt.zero_grad()
            obj = value()
            obj.backward()
            return obj

        try:
            opt.step(objective)
        except RuntimeError:
            continue
        with torch.no_grad():
            val = float(value())
        if math.isfinite(val) and (best is None or val < best[0]):
            best = (val, theta.detach().clone())
    a, b, e, al, be = best[1].tolist()
    return {"E": math.exp(e), "A": math.exp(a), "B": math.exp(b), "alpha": al, "beta": be, "objective": best[0]}


# ---------------------------------------------------------------- inference-optimal N vs V
def inference_error(n, v, a_coef, b_coef, alpha, beta, floor=0.0):
    return a_coef / np.power(n, alpha) + b_coef / np.power(v, beta) + floor


def inference_frontier(budget: float, q: int, v_values, a_coef, b_coef, alpha, beta,
                       n_min: float = 0.5, n_max: float = 72.0) -> list[dict]:
    """For each visual-token count V, the LLM size that spends `budget` = N (Q + V) and its error.

    N is in billions of parameters. Sizes outside [n_min, n_max] are marked unavailable.
    """
    rows = []
    for v in v_values:
        n = budget / (q + v)
        rows.append({"V": int(v), "N": n, "available": n_min <= n <= n_max,
                     "error": float(inference_error(n, v, a_coef, b_coef, alpha, beta))})
    return rows


# ---------------------------------------------------------------- modality interaction
def interaction(n, d, c_benefit, a_coef, alpha, b_coef, beta):
    """Aghajanyan et al.'s interaction term. Positive: sharing weights hurts. Negative: it helps."""
    return -c_benefit + a_coef / np.power(n, alpha) + b_coef / np.power(d, beta)


def crossover_size(d, c_benefit, a_coef, alpha, b_coef, beta) -> float:
    """The model size at which the interaction term reaches zero for a fixed data size d."""
    rest = c_benefit - b_coef / d**beta
    if rest <= 0:
        return math.inf
    return (a_coef / rest) ** (1.0 / alpha)


# ---------------------------------------------------------------- utility decay with repetition
def decayed_error(samples_seen, pool_size: float, a: float, b: float, tau: float):
    """Error after `samples_seen` views from a pool of `pool_size` unique samples.

    During the first pass, error = a * n^(-b). Each later pass k (k = 1, 2, ...) continues the curve
    with the smaller exponent b * (1/2)^(k / tau), so repeats keep helping but less each time.
    """
    out = []
    for n in np.atleast_1d(samples_seen):
        err = a * min(n, pool_size) ** (-b)
        k, start = 1, pool_size
        while n > start:
            end = min(n, start + pool_size)
            err *= (end / start) ** (-b * 0.5 ** (k / tau))
            k, start = k + 1, end
        out.append(err)
    return np.array(out)
