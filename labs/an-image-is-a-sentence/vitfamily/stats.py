"""Mean attention distance: how far, in pixels, a head looks on average.

For one query patch i: sum_j a_ij * |c_i - c_j|, with c the patch centres. Averaged over query patches
and images. Special tokens (CLS, distillation) are dropped and each row is renormalised over patches.
"""

from __future__ import annotations

import torch


def patch_distances(side: int, patch_px: int) -> torch.Tensor:
    """(N, N) Euclidean distance between patch centres, in pixels."""
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    c = torch.stack([y.flatten(), x.flatten()], dim=-1).float() * patch_px
    return torch.cdist(c, c)


def mean_attention_distance(attn: torch.Tensor, side: int, patch_px: int, n_special: int = 0) -> torch.Tensor:
    """attn (B, heads, N_special + N, N_special + N) -> (heads,) mean distance in pixels."""
    a = attn[:, :, n_special:, n_special:]
    a = a / a.sum(dim=-1, keepdim=True)
    d = patch_distances(side, patch_px).to(a)
    return (a * d).sum(dim=-1).mean(dim=(0, 2))


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Mean row entropy in nats. A uniform row over N keys has entropy log N."""
    return -(attn * attn.clamp_min(1e-12).log()).sum(dim=-1).mean()
