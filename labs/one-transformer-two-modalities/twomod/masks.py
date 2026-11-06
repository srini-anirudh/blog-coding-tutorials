"""Who may look at whom. True means "query i may attend to key j".

The whole causal-vs-bidirectional difference is which of these the model passes to attention.
"""

from __future__ import annotations

import torch


def causal(n: int) -> torch.Tensor:
    """Lower triangle: position t sees positions 0..t. A language model."""
    return torch.ones(n, n, dtype=torch.bool).tril()


def bidirectional(n: int) -> torch.Tensor:
    """Everything sees everything. A ViT (or BERT)."""
    return torch.ones(n, n, dtype=torch.bool)


def prefix_lm(n: int, prefix: int) -> torch.Tensor:
    """A VLM-style block mask: the first `prefix` tokens (the image) see each other; the rest are causal."""
    m = causal(n)
    m[:prefix, :prefix] = True
    return m


def with_padding(mask: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Combine an (N, N) mask with a (B, N) padding mask (1 = real token) -> (B, N, N). Keys at PAD are hidden.

    Rows for PAD queries keep their own key so softmax never sees an all-masked row.
    """
    keep = keep.bool()
    m = mask[None] & keep[:, None, :]
    eye = torch.eye(mask.shape[-1], dtype=torch.bool)[None]
    return m | (eye & ~keep[:, :, None])


def render(mask: torch.Tensor) -> str:
    return "\n".join("".join("#" if v else "." for v in row) for row in mask.tolist())
