"""A randomly initialised Idefics3 with SmolVLM's module layout, for fast offline tests.

Image 32x32, patch 8 -> 16 patches -> pixel shuffle x2 -> 4 image tokens.
"""

from __future__ import annotations

import torch
from transformers import Idefics3Config, Idefics3ForConditionalGeneration, LlamaConfig

IMAGE_TOKEN = 100
IMAGE_TOKENS_PER_IMAGE = 4


def tiny_vlm(seed: int = 0) -> Idefics3ForConditionalGeneration:
    torch.manual_seed(seed)
    text = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                       pad_token_id=0, tie_word_embeddings=False)
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  image_size=32, patch_size=8)
    cfg = Idefics3Config(vision_config=vision, text_config=text.to_dict(), scale_factor=2,
                         image_token_id=IMAGE_TOKEN, tie_word_embeddings=False)
    return Idefics3ForConditionalGeneration(cfg).eval()


def tiny_prompt(n_text: int = 3, seed: int = 0) -> torch.Tensor:
    """(T_p,) prompt ids: a few text tokens, then the image placeholders, then one more token."""
    g = torch.Generator().manual_seed(seed)
    text = torch.randint(1, 90, (n_text,), generator=g)
    return torch.cat([text, torch.full((IMAGE_TOKENS_PER_IMAGE,), IMAGE_TOKEN), torch.tensor([5])])


def tiny_pixels(n_images: int = 1, seed: int = 0) -> torch.Tensor:
    """(I, 1, 3, 32, 32) non-zero pixels (all-zero images are treated as padding by Idefics3)."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_images, 1, 3, 32, 32, generator=g) + 0.1
