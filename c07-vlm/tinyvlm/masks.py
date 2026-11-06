"""Attention masks for multimodal sequences, built explicitly so we can look at them.

The language model needs, for every query position i and key position j, "may i attend to j?":
    causal    j <= i
    padding   j is a real token (attention_mask[j] == 1)
    optional  image tokens of the same picture see each other in both directions (prefix-LM style,
              as in PaliGemma); the default decoder-only VLM keeps them causal like text.

Shapes: attention_mask B x N, image_index B x N  ->  allowed B x 1 x N x N (bool, broadcast over heads).
"""

from __future__ import annotations

import torch


def build_attention_mask(
    attention_mask: torch.Tensor, image_index: torch.Tensor | None = None, bidirectional_images: bool = False
) -> torch.Tensor:
    n = attention_mask.shape[-1]
    causal = torch.ones(n, n, dtype=torch.bool, device=attention_mask.device).tril()
    allowed = causal[None] & attention_mask.bool()[:, None, :]  # B x N x N: key must be real
    if bidirectional_images and image_index is not None:
        same_image = (image_index[:, :, None] == image_index[:, None, :]) & (image_index[:, :, None] >= 0)
        allowed = allowed | same_image
    return allowed[:, None]  # B x 1 x N x N


def to_additive(allowed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """bool mask -> additive float mask (0 where allowed, the dtype's minimum where blocked).

    Using finfo.min rather than -inf keeps fully padded query rows finite: softmax of a row of equal
    huge negatives is uniform, not NaN.
    """
    return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed, torch.finfo(dtype).min)


def render(allowed: torch.Tensor, labels: list[str]) -> str:
    """ASCII picture of one N x N mask: '#' = may attend, '.' = blocked."""
    rows = [f"{lab:>8} " + "".join("#" if a else "." for a in row) for lab, row in zip(labels, allowed.tolist())]
    return "\n".join(rows)
