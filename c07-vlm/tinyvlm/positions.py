"""Position ids for a sequence that mixes text and image tokens.

1D positions (what SmolLM2, LLaVA and SmolVLM use): every token, image or text, gets the next integer.
An image of 196 tokens therefore pushes the following text 196 positions further away.

M-RoPE style 3D positions (Qwen2-VL): each token gets (t, h, w). Text tokens use (p, p, p). Image tokens
share one t and spread over h and w by their grid cell, so a 14 x 14 image only advances text by 14.
"""

from __future__ import annotations

import torch


def position_ids_1d(attention_mask: torch.Tensor) -> torch.Tensor:
    """B x N mask -> B x N positions that count only real tokens.

    With left padding the first real token must still be position 0; padding slots get 1 (any value
    works, they are masked out) to match Hugging Face's convention.
    """
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(attention_mask == 0, 1)


def position_ids_3d(image_index: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """N image-index row (-1 = text) -> 3 x N (t, h, w) positions for crops laid out on a `grid` (rows, cols).

    Single sample, no padding: this is a teaching version of Qwen2-VL's get_rope_index.
    """
    rows, cols = grid
    pos = torch.zeros(3, image_index.shape[0], dtype=torch.long)
    next_pos, t = 0, 0
    while t < image_index.shape[0]:
        if image_index[t] < 0:
            pos[:, t] = next_pos
            next_pos += 1
            t += 1
            continue
        cells = torch.arange(rows * cols)
        span = slice(t, t + rows * cols)
        pos[0, span] = next_pos  # one temporal index for the whole picture
        pos[1, span] = next_pos + cells // cols  # row
        pos[2, span] = next_pos + cells % cols  # column
        next_pos += max(rows, cols)  # text resumes after the largest spatial extent
        t += rows * cols
    return pos
