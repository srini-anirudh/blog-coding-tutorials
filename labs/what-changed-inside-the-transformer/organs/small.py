"""Small changes that aren't small: chunked cross-entropy and attention sinks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_loss_sum(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy((h @ w.T).float(), y, reduction="sum")


def chunked_cross_entropy(hidden: torch.Tensor, head_weight: torch.Tensor, targets: torch.Tensor,
                          chunk: int) -> torch.Tensor:
    """Mean cross-entropy over (N, d) hidden states without ever holding all (N, V) logits.

    Each chunk's logits are built, reduced to a loss and freed. Activation checkpointing rebuilds them one chunk
    at a time in the backward pass, so the backward never holds them all either.
    """
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, hidden.shape[0], chunk):
        total = total + checkpoint(_chunk_loss_sum, hidden[i : i + chunk], head_weight, targets[i : i + chunk],
                                   use_reentrant=False)
    return total / hidden.shape[0]


def softmax_with_sink(logits: torch.Tensor, sink: torch.Tensor | float) -> torch.Tensor:
    """a_n = exp(z_n) / (exp(sigma) + sum_j exp(z_j)), as in gpt-oss. The weights sum to less than one."""
    sink = torch.as_tensor(sink, dtype=logits.dtype).expand(*logits.shape[:-1], 1)
    return torch.cat([logits, sink], -1).softmax(-1)[..., :-1]
