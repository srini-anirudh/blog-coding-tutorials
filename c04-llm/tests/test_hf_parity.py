"""The strongest test: Hugging Face's LlamaForCausalLM, given our weights, must agree with us."""

import pytest
import torch

from c04_helpers import random_ids, small_model
from tinyllm.hf import from_hf, from_pretrained, save_hf, to_hf


@pytest.mark.parametrize("overrides", [{}, {"tie_word_embeddings": False}, {"num_kv_heads": 1}, {"rope_theta": 500_000.0}])
def test_logits_and_loss_match_llama(overrides):
    model = small_model(**overrides)
    hf = to_hf(model)
    ids = random_ids(2, 33)
    ours = model(ids, labels=ids)
    with torch.no_grad():
        theirs = hf(ids, labels=ids)
    torch.testing.assert_close(ours.logits, theirs.logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ours.loss, theirs.loss, atol=1e-5, rtol=1e-5)


def test_greedy_generation_matches_hf_generate():
    model = small_model()
    hf = to_hf(model)
    prompt = random_ids(2, 6)
    ours = model.generate(prompt, max_new_tokens=25)
    theirs = hf.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=25, do_sample=False)
    assert torch.equal(ours, theirs)


def test_round_trip_through_hf():
    model = small_model()
    back = from_hf(to_hf(model))
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), back.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)


def test_save_hf_loads_with_automodel(tmp_path):
    from transformers import AutoModelForCausalLM

    model = small_model()
    save_hf(model, tmp_path)
    hf = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    ids = random_ids(1, 10)
    with torch.no_grad():
        torch.testing.assert_close(hf(ids).logits, model(ids).logits, atol=1e-5, rtol=1e-5)


@pytest.mark.network
@pytest.mark.slow
def test_smollm2_135m_logits_match():
    """Load a real pretrained checkpoint into TinyLLM and compare with transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    model = from_pretrained(name)
    assert model.num_parameters() == 134_515_008
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, hf(ids).logits, atol=1e-4, rtol=1e-4)
    out = model.generate(ids, max_new_tokens=8)
    ref = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, do_sample=False)
    assert torch.equal(out, ref)
