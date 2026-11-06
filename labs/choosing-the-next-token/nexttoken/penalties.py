"""Degeneration controls: logit edits that depend on what has already been generated.

Each function takes logits (B, V) and the token history (B, T) and returns new logits. None of them changes what the
model believes. They change the distribution the decoder samples from.
"""

from __future__ import annotations

import torch


def _seen(history: torch.Tensor, vocab: int) -> torch.Tensor:
    """(B, V) counts of each token in the history."""
    counts = torch.zeros(history.shape[0], vocab, dtype=torch.float32, device=history.device)
    return counts.scatter_add_(1, history, torch.ones_like(history, dtype=torch.float32))


def repetition_penalty(logits: torch.Tensor, history: torch.Tensor, theta: float) -> torch.Tensor:
    """CTRL's penalty with the sign branch: z/theta if z > 0, z*theta if z <= 0, for every seen token."""
    seen = _seen(history, logits.shape[-1]) > 0
    z = logits.float()
    penalized = torch.where(z > 0, z / theta, z * theta)
    return torch.where(seen, penalized, z)


def repetition_penalty_naive(logits: torch.Tensor, history: torch.Tensor, theta: float) -> torch.Tensor:
    """The sign trap: always divide. A negative logit moves toward zero, so the 'penalty' raises its probability."""
    seen = _seen(history, logits.shape[-1]) > 0
    z = logits.float()
    return torch.where(seen, z / theta, z)


def frequency_penalty(logits: torch.Tensor, history: torch.Tensor, alpha: float) -> torch.Tensor:
    """z_i - alpha * count(i). Additive, so shift-invariant."""
    return logits.float() - alpha * _seen(history, logits.shape[-1])


def presence_penalty(logits: torch.Tensor, history: torch.Tensor, beta: float) -> torch.Tensor:
    """z_i - beta * 1[count(i) > 0]. Pays once per token."""
    return logits.float() - beta * (_seen(history, logits.shape[-1]) > 0).float()


def no_repeat_ngram(logits: torch.Tensor, history: torch.Tensor, n: int) -> torch.Tensor:
    """Hard mask: any token that would complete an n-gram already in the history gets -inf."""
    z = logits.float().clone()
    if history.shape[1] < n - 1:
        return z
    for b in range(history.shape[0]):
        h = history[b].tolist()
        prefix = tuple(h[len(h) - (n - 1):]) if n > 1 else ()
        banned = [h[i + n - 1] for i in range(len(h) - n + 1) if tuple(h[i:i + n - 1]) == prefix]
        if banned:
            z[b, banned] = float("-inf")
    return z


def logit_bias(logits: torch.Tensor, bias: dict[int, float]) -> torch.Tensor:
    """Add a constant per token id. -inf bans the token."""
    z = logits.float().clone()
    for i, v in bias.items():
        z[:, i] += v
    return z
