"""Our loops against model.generate and the library's logits processors."""

import pytest
import torch
from transformers import LlamaForCausalLM, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

from hfstack.generation import generate_with_cache, greedy_no_cache, process_logits
from hfstack.utils import tiny_llama_config


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return LlamaForCausalLM(tiny_llama_config()).eval()


def test_cache_equals_no_cache(model):
    ids = torch.randint(3, 512, (2, 6))
    assert torch.equal(greedy_no_cache(model, ids, 10), generate_with_cache(model, ids, max_new_tokens=10))


def test_greedy_equals_generate_with_left_padding(model):
    ids = torch.randint(3, 512, (2, 6))
    ids[0, :2] = 0
    mask = (ids != 0).long()
    ours = generate_with_cache(model, ids, mask, max_new_tokens=8, eos_token_id=2, pad_token_id=0)
    theirs = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8, do_sample=False, eos_token_id=2, pad_token_id=0)
    assert torch.equal(ours, theirs)


def test_sampling_equals_generate_with_same_seed(model):
    ids = torch.randint(3, 512, (2, 6))
    knobs = dict(temperature=0.7, top_k=50, top_p=0.9)
    torch.manual_seed(1)
    ours = generate_with_cache(model, ids, max_new_tokens=8, do_sample=True, eos_token_id=2, pad_token_id=0, **knobs)
    torch.manual_seed(1)
    theirs = model.generate(input_ids=ids, max_new_tokens=8, do_sample=True, eos_token_id=2, pad_token_id=0, **knobs)
    assert torch.equal(ours, theirs)


@pytest.mark.parametrize("temperature,top_k,top_p", [(0.7, 0, 1.0), (1.0, 5, 1.0), (1.0, 0, 0.9), (0.5, 20, 0.8)])
def test_processors_equal_library(temperature, top_k, top_p):
    torch.manual_seed(0)
    logits = torch.randn(3, 100)
    expected = logits.clone()
    ids = torch.zeros(3, 1, dtype=torch.long)
    if temperature != 1.0:
        expected = TemperatureLogitsWarper(temperature)(ids, expected)
    if top_k:
        expected = TopKLogitsWarper(top_k)(ids, expected)
    if top_p < 1.0:
        expected = TopPLogitsWarper(top_p)(ids, expected)
    assert torch.equal(process_logits(logits, temperature, top_k, top_p), expected)
