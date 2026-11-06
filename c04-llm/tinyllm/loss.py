"""Next-token cross-entropy: position t predicts token t + 1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Mean cross-entropy over the positions that have a real next token.

    logits: (B, T, V)   labels: (B, T), usually the input ids themselves.
    The shift happens here, as in Hugging Face: logits[:, :-1] are scored against labels[:, 1:].
    Positions whose label is ignore_index (padding, prompt tokens) do not count.
    """
    shift_logits = logits[:, :-1, :].float()  # (B, T-1, V); float32 so log-softmax is exact enough
    shift_labels = labels[:, 1:]  # (B, T-1)
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
    )
