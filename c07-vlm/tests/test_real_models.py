"""Checks against the real checkpoints (downloaded once, then read from the local Hub cache)."""

import numpy as np
import pytest
import torch
from PIL import Image

from tinyvlm import IGNORE_INDEX, LM_ID, VISION_ID, TinyVLM, TinyVLMProcessor, build_tinyvlm

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def built():
    return build_tinyvlm()


def photo() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8))


def test_chat_template_matches_smollm2(built):
    _, proc = built
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(LM_ID)
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"},
                {"role": "assistant", "content": "Hello."}]
    assert proc.apply_chat_template(messages) == hf.apply_chat_template(messages, tokenize=False)
    assert proc.apply_chat_template(messages[:2], True) == hf.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)


def test_piecewise_tokenisation_equals_whole_string(built):
    """Labels are built piece by piece; the ids must be the same as tokenising the full string."""
    _, proc = built
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe it."}]},
                {"role": "assistant", "content": "Two red circles on a white background."}]
    ex = proc.build_example(messages, [photo()])
    full = proc._expand(proc.apply_chat_template(messages), iter(ex["num_crops"]))
    assert ex["input_ids"] == proc.tokenizer(full, add_special_tokens=False)["input_ids"]
    supervised = [t for t, lab in zip(ex["input_ids"], ex["labels"]) if lab != IGNORE_INDEX]
    assert proc.tokenizer.decode(supervised) == "Two red circles on a white background.<|im_end|>"


def test_vision_tower_matches_full_siglip(built):
    model, proc = built
    from transformers import SiglipModel

    full = SiglipModel.from_pretrained(VISION_ID).eval()
    pixels = proc.image_processor([photo()])["pixel_values"]
    with torch.no_grad():
        ours = model.vision(pixel_values=pixels).last_hidden_state
        ref = full.vision_model(pixel_values=pixels).last_hidden_state
    assert ours.shape == (1, 196, 768)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_text_only_logits_match_smollm2(built):
    model, proc = built
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.float32).eval()
    ids = proc.tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        ours, theirs = model(input_ids=ids).logits, ref(input_ids=ids).logits
    assert torch.allclose(ours[..., : theirs.shape[-1]], theirs, atol=1e-4)  # original 49152 columns


def test_shapes_through_the_real_model(built):
    model, proc = built
    prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}], True)
    batch = proc(prompt, [photo()])
    assert (batch["input_ids"] == model.config.image_token_id).sum() == 49  # 196 patches, pixel shuffle 2
    with torch.no_grad():
        features = model.get_image_features(batch["pixel_values"])
        logits = model(batch["input_ids"], batch["pixel_values"], batch["attention_mask"]).logits
    assert features.shape == (1, 49, 576)
    assert logits.shape == (1, batch["input_ids"].shape[1], 49216)


@pytest.mark.slow
def test_full_save_load_round_trip(built, tmp_path):
    model, proc = built
    model.save_pretrained(tmp_path)
    proc.save_pretrained(tmp_path)
    again = TinyVLM.from_pretrained(tmp_path).eval()
    proc2 = TinyVLMProcessor.from_pretrained(tmp_path)
    prompt = proc2.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}], True)
    batch = proc2(prompt, [photo()])
    with torch.no_grad():
        a = model(batch["input_ids"], batch["pixel_values"], batch["attention_mask"]).logits
        b = again(batch["input_ids"], batch["pixel_values"], batch["attention_mask"]).logits
    assert torch.equal(a, b)
