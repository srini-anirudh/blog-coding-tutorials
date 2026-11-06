"""Put image features where the <image> placeholders are.

The processor leaves N_img placeholder ids per crop in `input_ids`. The language model embeds them
like any token (a meaningless vector), and we overwrite those rows with the projected image features.

    text_embeds     B x N x d_lm    embeddings of every token, placeholders included
    input_ids       B x N           placeholder positions are input_ids == image_token_id
    image_features  n x N_img x d_lm  n crops in the order their placeholders appear (row-major over B, N)
"""

from __future__ import annotations

import torch


def check_counts(input_ids: torch.Tensor, image_features: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Return the placeholder mask (B x N, bool) after checking it matches the number of features."""
    mask = input_ids == image_token_id
    n_slots, n_features = int(mask.sum()), image_features.shape[0] * image_features.shape[1]
    if n_slots != n_features:
        raise ValueError(
            f"{n_slots} <image> placeholders but {n_features} image feature vectors "
            f"({image_features.shape[0]} crops x {image_features.shape[1]} tokens). "
            "The processor and the model disagree about tokens per crop."
        )
    return mask


def merge_naive(
    text_embeds: torch.Tensor, input_ids: torch.Tensor, image_features: torch.Tensor, image_token_id: int
) -> torch.Tensor:
    """Reference version: walk the sequence and copy one feature vector per placeholder."""
    check_counts(input_ids, image_features, image_token_id)
    out = text_embeds.clone()
    flat = image_features.reshape(-1, image_features.shape[-1])  # (n * N_img) x d_lm, in order
    k = 0
    for b in range(input_ids.shape[0]):
        for t in range(input_ids.shape[1]):
            if input_ids[b, t] == image_token_id:
                out[b, t] = flat[k]
                k += 1
    return out


def merge_masked_scatter(
    text_embeds: torch.Tensor, input_ids: torch.Tensor, image_features: torch.Tensor, image_token_id: int
) -> torch.Tensor:
    """Fast version, same result: one masked_scatter fills the masked rows in row-major order."""
    mask = check_counts(input_ids, image_features, image_token_id)
    mask = mask.unsqueeze(-1).expand_as(text_embeds)  # B x N x d_lm
    return text_embeds.masked_scatter(mask, image_features.to(text_embeds.dtype))
