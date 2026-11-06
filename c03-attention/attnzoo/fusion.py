"""Three ways to let text read an image.

Early fusion:      [image tokens][text tokens] -> one self-attention (LLaVA, Qwen-VL)
Cross-attention:   text queries -> image keys/values in extra layers (Llama 3.2 Vision)
Gated cross-attn:  the same, behind tanh gates that start closed (Flamingo, see cross.py)
"""

from __future__ import annotations

import torch
from torch import nn

from .config import AttentionConfig
from .cross import CrossAttention, GatedCrossAttentionBlock  # noqa: F401  (re-exported)
from .gqa import GroupedQueryAttention
from .masks import multimodal_mask


def early_fusion_sequence(image_tokens: torch.Tensor, text_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """image (B, Nv, D), text (B, Nt, D) -> sequence (B, Nv + Nt, D), mask (B, 1, N, N).

    The image block is bidirectional, the text is causal and sees the whole image.
    """
    b, nv, _ = image_tokens.shape
    nt = text_tokens.size(1)
    span_ids = torch.cat([torch.ones(b, nv, dtype=torch.long), torch.zeros(b, nt, dtype=torch.long)], dim=1)
    return torch.cat([image_tokens, text_tokens], dim=1), multimodal_mask(span_ids.to(image_tokens.device))


class EarlyFusionBlock(nn.Module):
    """One self-attention over the joint sequence. The mask, not the module, encodes the modalities."""

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        # causal=False: the multimodal mask passed in already contains the causal part for text.
        self.attn = GroupedQueryAttention(AttentionConfig(**{**config.to_dict(), "causal": False}))
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        seq, mask = early_fusion_sequence(image_tokens, text_tokens)
        return seq + self.attn(self.norm(seq), attention_mask=mask)


class CrossAttentionFusionBlock(nn.Module):
    """Text self-attends causally, then reads the image through cross-attention. Image tokens never enter the text sequence."""

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.self_attn = GroupedQueryAttention(config)
        self.cross_attn = CrossAttention(AttentionConfig(**{**config.to_dict(), "positional_encoding": "none"}))
        self.norm1, self.norm2 = nn.LayerNorm(config.hidden_size), nn.LayerNorm(config.hidden_size)

    def forward(self, text_tokens: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        x = text_tokens + self.self_attn(self.norm1(text_tokens))
        return x + self.cross_attn(self.norm2(x), image_tokens)
