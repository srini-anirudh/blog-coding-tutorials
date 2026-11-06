"""The custom VLM: nested config, placeholder scatter, misalignment error, round trip, generate."""

import json

import pytest
import torch
from transformers import AutoModelForImageTextToText

from hfstack import custom_model, custom_vlm
from hfstack.custom_vlm import BackwardPassVLMConfig, BackwardPassVLMForConditionalGeneration, expand_image_tokens

PATCHES = 16


@pytest.fixture(scope="module")
def model():
    custom_model.register()
    custom_vlm.register()
    torch.manual_seed(0)
    config = BackwardPassVLMConfig(
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=8),
        text_config=dict(vocab_size=512),
    )
    return BackwardPassVLMForConditionalGeneration(config).eval()


def batch():
    ids = torch.tensor([expand_image_tokens([1, 3, 50, 60], 3, PATCHES)])
    return ids, torch.randn(1, 3, 32, 32)


def test_expand_image_tokens():
    assert expand_image_tokens([1, 3, 5], 3, 2) == [1, 3, 3, 5]


def test_forward_shapes(model):
    ids, pixels = batch()
    assert model.image_features(pixels).shape == (1, PATCHES, 64)
    assert model(input_ids=ids, pixel_values=pixels).logits.shape == (1, ids.size(1), 512)


def test_image_changes_the_prediction(model):
    ids, pixels = batch()
    a = model(input_ids=ids, pixel_values=pixels).logits
    b = model(input_ids=ids, pixel_values=pixels + 1).logits
    assert not torch.allclose(a[0, -1], b[0, -1])


def test_misaligned_image_tokens_raise(model):
    _, pixels = batch()
    with pytest.raises(ValueError, match="image tokens"):
        model(input_ids=torch.tensor([[1, 3, 50]]), pixel_values=pixels)


def test_round_trip(model, tmp_path):
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["vision_config"]["model_type"] == "siglip_vision_model"
    assert saved["text_config"]["model_type"] == "backward_pass"
    loaded = AutoModelForImageTextToText.from_pretrained(tmp_path).eval()
    ids, pixels = batch()
    assert torch.allclose(loaded(input_ids=ids, pixel_values=pixels).logits, model(input_ids=ids, pixel_values=pixels).logits)


def test_generate_with_cache_matches_full_recompute(model):
    ids, pixels = batch()
    tokens = model.generate(input_ids=ids, pixel_values=pixels, max_new_tokens=4, do_sample=False)
    current = ids
    for _ in range(4):
        # Recompute from scratch each step. Only the placeholders receive image features.
        logits = model(input_ids=current, pixel_values=pixels, use_cache=False).logits
        current = torch.cat([current, logits[:, -1].argmax(-1, keepdim=True)], dim=-1)
    assert torch.equal(tokens, current)
