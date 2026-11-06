"""Past softmax: a fixed-size state as an associative memory. Additive (linear attention) vs delta-rule writes,
against softmax attention over a full KV cache.

The state S is d_v x d_k. Write a pair (k, v); read with a query q as S q.
    linear:  S <- S + v k^T
    delta:   S <- S (I - beta k k^T) + beta v k^T                 (DeltaNet)
    gated:   S <- alpha S (I - beta k k^T) + beta v k^T           (Gated DeltaNet, Yang et al. 2024)
The post writes the gated update as S (alpha I - beta k k^T) + beta v k^T. The two agree when alpha = 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def write_linear(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return values.T @ keys  # sum_n v_n k_n^T, (d_v, d_k)


def write_delta(keys: torch.Tensor, values: torch.Tensor, beta: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    d_k, d_v = keys.shape[1], values.shape[1]
    s = keys.new_zeros(d_v, d_k)
    for k, v in zip(keys, values):
        pred = s @ k  # what the state currently returns for this key
        s = alpha * (s - beta * torch.outer(pred, k)) + beta * torch.outer(v, k)
    return s


def read_state(s: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    return queries @ s.T


def read_softmax(keys: torch.Tensor, values: torch.Tensor, queries: torch.Tensor, scale: float) -> torch.Tensor:
    return (scale * queries @ keys.T).softmax(-1) @ values


def recall_task(n: int, d: int, seed: int, shared: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """n random unit keys and n random unit values in d dimensions.

    shared > 0 adds a common direction to every key. shared = 1 gives a mean key-key cosine of about 0.5,
    the case where an additive state suffers most.
    """
    g = torch.Generator().manual_seed(seed)
    common = F.normalize(torch.randn(d, generator=g, dtype=torch.float64), dim=0)
    raw = torch.randn(n, d, generator=g, dtype=torch.float64) / d**0.5 + shared * common
    keys = F.normalize(raw, dim=-1)
    values = F.normalize(torch.randn(n, d, generator=g, dtype=torch.float64), dim=-1)
    return keys, values


def recall_cosine(pred: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between each retrieved vector and the value that was stored under its key."""
    return F.cosine_similarity(pred, values, dim=-1)
