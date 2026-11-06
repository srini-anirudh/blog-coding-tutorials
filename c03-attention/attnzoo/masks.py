"""Attention masks, built by hand.

Convention used everywhere in attnzoo (and by torch SDPA for bool masks):
    True  = this query may attend to this key
    False = blocked
A mask has shape (Nq, Nk) or (B, 1, Nq, Nk); broadcasting adds the head dim.
"""

from __future__ import annotations

import torch


def bidirectional_mask(n: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Every token sees every token (encoders, ViT). Shape (n, n)."""
    return torch.ones(n, n, dtype=torch.bool, device=device)


def causal_mask(nq: int, nk: int | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Lower-triangular mask, aligned to the bottom-right corner. Shape (nq, nk).

    With a KV cache the queries are the *last* nq of nk positions, so query i
    sits at absolute position i + (nk - nq) and may see keys up to that position.
    """
    nk = nq if nk is None else nk
    offset = nk - nq
    q_pos = torch.arange(nq, device=device)[:, None] + offset
    k_pos = torch.arange(nk, device=device)[None, :]
    return k_pos <= q_pos


def sliding_window_mask(
    nq: int, window: int, nk: int | None = None, device: torch.device | str | None = None
) -> torch.Tensor:
    """Causal mask that also forgets keys more than `window - 1` steps back. Shape (nq, nk).

    Each query sees exactly the last `window` positions, itself included.
    """
    nk = nq if nk is None else nk
    q_pos = torch.arange(nq, device=device)[:, None] + (nk - nq)
    k_pos = torch.arange(nk, device=device)[None, :]
    return (k_pos <= q_pos) & (k_pos > q_pos - window)


def padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Turn a tokenizer-style (B, Nk) 1/0 mask into a (B, 1, 1, Nk) key mask.

    Padding is a property of keys: no query should read a pad token.
    """
    return attention_mask.bool()[:, None, None, :]


def combine(*masks: torch.Tensor | None) -> torch.Tensor | None:
    """Logical AND of masks with broadcasting; None entries are ignored."""
    out = None
    for m in masks:
        if m is None:
            continue
        out = m if out is None else out & m
    return out


def prefix_lm_mask(n: int, prefix_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Prefix-LM: the first `prefix_len` tokens see each other freely, the rest is causal. Shape (n, n)."""
    mask = causal_mask(n, device=device)
    mask[:prefix_len, :prefix_len] = True
    return mask


def document_mask(doc_ids: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Block-diagonal mask for packed sequences: tokens only see their own document.

    doc_ids: (B, N) integer document index per token -> (B, 1, N, N)
    """
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    if causal:
        same_doc = same_doc & causal_mask(doc_ids.size(1), device=doc_ids.device)
    return same_doc[:, None]


def multimodal_mask(span_ids: torch.Tensor) -> torch.Tensor:
    """Causal everywhere, but tokens of one image see each other in both directions.

    span_ids: (B, N) with 0 for text tokens and k > 0 for tokens of the k-th image.
    Returns (B, 1, N, N). Text after an image sees the whole image (it is in the past);
    image patches see the full image they belong to, as most VLM decoders do.
    """
    n = span_ids.size(1)
    causal = causal_mask(n, device=span_ids.device)
    same_image = (span_ids[:, :, None] == span_ids[:, None, :]) & (span_ids[:, :, None] > 0)
    return (causal | same_image)[:, None]


def local_global_mask(
    n: int, window: int, global_positions: list[int], causal: bool = False, device: torch.device | str | None = None
) -> torch.Tensor:
    """Longformer-style mask: a local band plus a few global tokens. Shape (n, n).

    Global tokens attend to everything and everything attends to them.
    """
    pos = torch.arange(n, device=device)
    band = (pos[None, :] - pos[:, None]).abs() < window
    is_global = torch.zeros(n, dtype=torch.bool, device=device)
    is_global[global_positions] = True
    mask = band | is_global[:, None] | is_global[None, :]
    if causal:
        mask = mask & causal_mask(n, device=device)
    return mask


def expand_block_layout(layout: torch.Tensor, block_size: int) -> torch.Tensor:
    """Turn a (nb, nb) block-level layout into a (nb*bs, nb*bs) token mask."""
    return layout.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)


def block_sparse_layout(num_blocks: int, local_blocks: int = 1, global_blocks: int = 1, causal: bool = True) -> torch.Tensor:
    """A pedagogical BigBird-like block layout: local diagonal band + leading global blocks."""
    b = torch.arange(num_blocks)
    layout = (b[:, None] - b[None, :]).abs() < local_blocks
    layout[:, :global_blocks] = True
    layout[:global_blocks, :] = True
    if causal:
        layout = layout & (b[None, :] <= b[:, None])
    return layout


def structural_mask(
    nq: int, nk: int, causal: bool, window_size: int | None, device: torch.device | str | None = None
) -> torch.Tensor | None:
    """The mask implied by the architecture flags, or None when nothing is blocked."""
    if window_size is not None:
        return sliding_window_mask(nq, window_size, nk, device=device)
    if causal:
        return causal_mask(nq, nk, device=device)
    return None


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """bool mask -> additive bias: 0 where allowed, -inf where blocked."""
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return bias.masked_fill(~mask, float("-inf"))


def render(mask: torch.Tensor) -> str:
    """ASCII picture of a 2D mask: '#' attends, '.' blocked. Rows are queries."""
    return "\n".join("".join("#" if x else "." for x in row) for row in mask.tolist())
