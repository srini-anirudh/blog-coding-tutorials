"""Linear attention keeps a d x d summary. Softmax attention keeps every token. What does recall cost?

Associative recall, with no training: store n (key, value) pairs, then query with one stored key and
ask which value comes back. Keys are random unit vectors in R^d. Values are random unit vectors too,
and the answer is read out by nearest stored value.

  softmax   z = softmax(beta q K^T) V                     cache: n keys + n values, grows with n
  linear    z = q^T S, S = sum_j k_j v_j^T                state: d x d, fixed
  elu+1     z = phi(q)^T S / phi(q)^T u, phi = elu + 1    state: d x d plus d (Katharopoulos et al.)

For the linear state, z = v_j + sum_{i != j} (k_i . k_j) v_i. Each cross term has size about 1/sqrt(d),
so the noise grows with n while the signal stays at 1, and recall fails once n passes a small multiple
of d. A positive feature map makes every cross term positive, so the noise does not average out:
phi(k) . phi(k') is large for every pair, and the read returns something close to the mean value.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_pairs(n: int, d: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    k = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    v = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    return k, v


def recall_softmax(k: torch.Tensor, v: torch.Tensor, beta: float = 32.0) -> torch.Tensor:
    return torch.softmax(beta * k @ k.T, dim=-1) @ v  # every stored key is queried once


def linear_state(k: torch.Tensor, v: torch.Tensor, phi=None) -> tuple[torch.Tensor, torch.Tensor]:
    """The running sums, built one token at a time as the recurrence would: S_t = S_{t-1} + phi(k_t) v_t^T."""
    fk = k if phi is None else phi(k)
    s = torch.zeros(fk.size(-1), v.size(-1))
    u = torch.zeros(fk.size(-1))
    for t in range(k.size(0)):
        s = s + fk[t, :, None] * v[t, None, :]
        u = u + fk[t]
    return s, u


def recall_linear(k: torch.Tensor, v: torch.Tensor, phi=None) -> torch.Tensor:
    s, u = linear_state(k, v, phi)
    if phi is None:
        return k @ s
    fq = phi(k)
    return (fq @ s) / (fq @ u)[:, None]


def elu1(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


def elu1_sharp(x: torch.Tensor) -> torch.Tensor:
    """elu + 1 on keys scaled by 32: a sharper positive map, as a learned projection could make."""
    return F.elu(32.0 * x) + 1.0


def accuracy(z: torch.Tensor, v: torch.Tensor) -> float:
    """Fraction of queries whose output is closest (by cosine) to the right stored value."""
    return (F.normalize(z, dim=-1) @ v.T).argmax(-1).eq(torch.arange(v.size(0))).float().mean().item()


def recall_table(d: int, ns: list[int], seeds: int = 3) -> list[dict]:
    rows = []
    for n in ns:
        acc = {"softmax": 0.0, "linear": 0.0, "elu+1": 0.0, "elu+1 (32k)": 0.0}
        for seed in range(seeds):
            k, v = make_pairs(n, d, seed)
            acc["softmax"] += accuracy(recall_softmax(k, v), v) / seeds
            acc["linear"] += accuracy(recall_linear(k, v), v) / seeds
            acc["elu+1"] += accuracy(recall_linear(k, v, elu1), v) / seeds
            acc["elu+1 (32k)"] += accuracy(recall_linear(k, v, elu1_sharp), v) / seeds
        rows.append({
            "n": n, **acc,
            "softmax floats": 2 * n * d,  # the cache: n keys and n values
            "linear floats": d * d + d,  # S and u, whatever n is
        })
    return rows


def capacity(d: int, threshold: float = 0.9, seeds: int = 3, phi=None) -> int:
    """Largest n on a grid of d/8, 2d/8, 3d/8, ... where linear recall stays at or above `threshold`."""
    best = 0
    for n in range(max(d // 8, 1), 8 * d + 1, max(d // 8, 1)):
        acc = 0.0
        for s in range(seeds):
            k, v = make_pairs(n, d, s)
            acc += accuracy(recall_linear(k, v, phi), v) / seeds
        if acc < threshold:
            break
        best = n
    return best
