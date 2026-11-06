"""A randomly initialised Qwen2-VL at toy size and hand-built inputs, so tests need no download."""

from __future__ import annotations

import torch

IMAGE_TOKEN, VSTART, VEND = 90, 91, 92


def tiny_qwen2vl(seed: int = 0):
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    torch.manual_seed(seed)
    config = Qwen2VLConfig(
        text_config=dict(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3]},
                         eos_token_id=99),
        vision_config=dict(depth=2, embed_dim=32, hidden_size=64, num_heads=4, patch_size=14,
                           spatial_merge_size=2, temporal_patch_size=2, in_chans=3, mlp_ratio=2),
        image_token_id=IMAGE_TOKEN, vision_start_token_id=VSTART, vision_end_token_id=VEND)
    config._attn_implementation = "sdpa"
    return Qwen2VLForConditionalGeneration(config).eval()


def tiny_inputs(rows: int = 4, cols: int = 6, before: int = 3, after: int = 5, seed: int = 0) -> dict:
    """input_ids with rows x cols visual tokens (a 2rows x 2cols patch grid) between text tokens."""
    g = torch.Generator().manual_seed(seed)
    v = rows * cols
    ids = torch.cat([torch.randint(0, 80, (before,), generator=g), torch.tensor([VSTART]),
                     torch.full((v,), IMAGE_TOKEN), torch.tensor([VEND]),
                     torch.randint(0, 80, (after,), generator=g)])[None]
    patches = (2 * rows) * (2 * cols)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": (ids == IMAGE_TOKEN).int(),
            "pixel_values": torch.randn(patches, 3 * 2 * 14 * 14, generator=g),
            "image_grid_thw": torch.tensor([[1, 2 * rows, 2 * cols]])}
