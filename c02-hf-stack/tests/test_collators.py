"""The collator's three jobs: common length, attention mask, -100 on everything not trained."""

import torch

from hfstack.collators import CausalLMCollator

IGNORE = -100


def test_shapes_mask_and_label_padding():
    features = [{"input_ids": [5, 6, 7]}, {"input_ids": [8]}]
    batch = CausalLMCollator(pad_token_id=0)(features)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[5, 6, 7], [8, IGNORE, IGNORE]]


def test_prompt_labels_survive():
    features = [{"input_ids": [5, 6, 7], "labels": [IGNORE, IGNORE, 7]}, {"input_ids": [8, 9], "labels": [8, 9]}]
    batch = CausalLMCollator(pad_token_id=0)(features)
    assert batch["labels"].tolist() == [[IGNORE, IGNORE, 7], [8, 9, IGNORE]]


def test_pad_equal_to_eos_keeps_real_eos_labels():
    # Padding is by position, so a real EOS token with the same ID as the pad token stays a target.
    eos = 2
    batch = CausalLMCollator(pad_token_id=eos)([{"input_ids": [5, eos]}, {"input_ids": [6, 7, 8, eos]}])
    assert batch["labels"].tolist() == [[5, eos, IGNORE, IGNORE], [6, 7, 8, eos]]


def test_pad_to_multiple_of():
    batch = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=8)([{"input_ids": [1] * 9}])
    assert batch["input_ids"].shape == torch.Size([1, 16])
