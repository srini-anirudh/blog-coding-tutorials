"""Tests that download tokenizer and processor files (small, cached after the first run)."""

import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, DataCollatorForLanguageModeling

from hfstack.chat import assistant_labels_by_prefix, assistant_labels_by_template, chat_ids
from hfstack.collators import CausalLMCollator, VisionChatCollator
from hfstack.utils import BASE_LM, CHAT_LM, SMALL_VLM

pytestmark = pytest.mark.network

MESSAGES = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "6"},
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(CHAT_LM)


def test_encode_decode_round_trip(tok):
    text = "The backward pass computes gradients."
    assert tok.decode(tok(text)["input_ids"]) == text


def test_smollm2_tokenizer_adds_no_bos():
    t = AutoTokenizer.from_pretrained(BASE_LM)
    assert t("Hello")["input_ids"][0] != t.bos_token_id


def test_two_assistant_mask_methods_agree(tok):
    a = assistant_labels_by_prefix(tok, MESSAGES)
    b = assistant_labels_by_template(tok, MESSAGES)
    assert a == b


def test_only_assistant_text_and_end_of_turn_are_labelled(tok):
    example = assistant_labels_by_prefix(tok, MESSAGES)
    kept = [label for label in example["labels"] if label != -100]
    assert tok.decode(kept) == "4<|im_end|>6<|im_end|>"


def test_generation_prompt_is_a_prefix_of_the_full_text(tok):
    prompt = chat_ids(tok, MESSAGES[:2], add_generation_prompt=True)
    full = chat_ids(tok, MESSAGES[:3])
    assert full[: len(prompt)] == prompt


def test_library_collator_drops_real_eos_when_pad_is_eos(tok):
    # Documents the failure the post warns about. If this starts failing, the library changed.
    feature = {"input_ids": chat_ids(tok, MESSAGES[:3])}
    library = DataCollatorForLanguageModeling(tok, mlm=False)([feature])
    ours = CausalLMCollator(pad_token_id=tok.pad_token_id)([feature])
    eos = torch.tensor(feature["input_ids"]) == tok.eos_token_id
    assert (library["labels"][0][eos] == -100).all()
    assert (ours["labels"][0][eos] == tok.eos_token_id).all()


def test_processor_token_count_matches_vision_config():
    processor = AutoProcessor.from_pretrained(SMALL_VLM)
    config = AutoConfig.from_pretrained(SMALL_VLM)
    v = config.vision_config
    per_tile = (v.image_size // v.patch_size) ** 2 // config.scale_factor**2
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    batch = processor(text=[prompt], images=[[Image.new("RGB", (640, 480))]], return_tensors="pt")
    tiles = batch["pixel_values"].shape[1]
    assert (batch["input_ids"] == config.image_token_id).sum().item() == tiles * per_tile


def test_vision_collator_labels_only_the_reply():
    processor = AutoProcessor.from_pretrained(SMALL_VLM)
    user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}]
    reply = lambda text: [{"role": "assistant", "content": [{"type": "text", "text": text}]}]  # noqa: E731
    batch = VisionChatCollator(processor)(
        [
            {"image": Image.new("RGB", (300, 300)), "messages": user + reply("Seven")},
            {"image": Image.new("RGB", (640, 480)), "messages": user + reply("Two")},
        ]
    )
    assert batch["labels"].shape == batch["input_ids"].shape
    decoded = [processor.tokenizer.decode(row[row != -100]).strip() for row in batch["labels"]]
    assert decoded[0].startswith("Seven") and decoded[1].startswith("Two")
