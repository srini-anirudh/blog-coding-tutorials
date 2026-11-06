"""CUDA-only checks: on a GPU, the fused kernel must still agree with the reference attention.

FlashAttention accepts only bf16 and half, so these tests run the model in bf16.
Skipped when no GPU is visible. Run them with: python -m pytest c04-llm -m gpu
"""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from c04_helpers import random_ids, small_model

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]


@pytest.mark.parametrize("num_kv_heads", [4, 2, 1])
def test_flash_attention_matches_naive(num_kv_heads):
    model = small_model(num_kv_heads=num_kv_heads).to("cuda", dtype=torch.bfloat16)
    ids = random_ids(2, 32).to("cuda")
    # Naming one backend makes PyTorch raise, rather than fall back, if FlashAttention cannot run.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        flash = model(ids).logits.float()
    for layer in model.layers:
        layer.self_attn.backend = "naive"
    naive = model(ids).logits.float()
    assert (flash - naive).abs().max() < 0.05


def test_flash_attention_decode_step_matches_full_forward():
    model = small_model().to("cuda", dtype=torch.bfloat16)
    ids = random_ids(1, 16).to("cuda")
    full = model(ids).logits.float()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = model(ids[:, :15], use_cache=True)
        step = model(ids[:, 15:], past_key_values=out.past_key_values).logits.float()
    assert (step[:, 0] - full[:, -1]).abs().max() < 0.05


def test_float32_cache_matches_full_forward_on_gpu():
    model = small_model().to("cuda")
    ids = random_ids(1, 24).to("cuda")
    full = model(ids).logits
    cache = model(ids[:, :8], use_cache=True).past_key_values
    chunk = model(ids[:, 8:16], past_key_values=cache).logits  # chunked prefill: the shifted-mask path
    assert (chunk - full[:, 8:16]).abs().max() < 1e-4
