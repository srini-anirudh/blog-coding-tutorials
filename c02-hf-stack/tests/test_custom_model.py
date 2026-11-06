"""BackwardPass integration: Auto dispatch, save/load, cache, causality, Trainer contract."""

import json

import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from hfstack.collators import CausalLMCollator
from hfstack.custom_model import BackwardPassConfig, BackwardPassForCausalLM, BackwardPassModel, register
from hfstack.data import random_token_dataset
from hfstack.generation import generate_with_cache, greedy_no_cache
from hfstack.training import LoopConfig, clone_model, train_raw, train_with_trainer


@pytest.fixture(scope="module")
def model():
    register()
    torch.manual_seed(0)
    return BackwardPassForCausalLM(BackwardPassConfig()).eval()


def test_shapes(model):
    ids = torch.randint(3, 512, (2, 7))
    out = model(input_ids=ids, labels=ids, output_hidden_states=True)
    assert out.logits.shape == (2, 7, 512)
    assert out.loss.ndim == 0
    assert len(out.hidden_states) == model.config.num_hidden_layers + 1


def test_save_load_round_trip_through_auto_classes(model, tmp_path):
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["model_type"] == "backward_pass"
    assert saved["architectures"] == ["BackwardPassForCausalLM"]

    assert isinstance(AutoConfig.from_pretrained(tmp_path), BackwardPassConfig)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert isinstance(loaded, BackwardPassForCausalLM)
    ids = torch.randint(3, 512, (2, 7))
    assert torch.equal(model(input_ids=ids).logits, loaded(input_ids=ids).logits)
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight


def test_base_model_loads_from_causal_lm_checkpoint(model, tmp_path):
    model.save_pretrained(tmp_path)
    base = AutoModel.from_pretrained(tmp_path).eval()
    assert isinstance(base, BackwardPassModel)
    ids = torch.randint(3, 512, (1, 5))
    expected = model.model(input_ids=ids).last_hidden_state
    assert torch.equal(base(input_ids=ids).last_hidden_state, expected)


def test_causality(model):
    ids = torch.randint(3, 512, (1, 8))
    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 512
    a, b = model(input_ids=ids).logits, model(input_ids=changed).logits
    assert torch.equal(a[0, :-1], b[0, :-1])
    assert not torch.equal(a[0, -1], b[0, -1])


def test_cached_forward_equals_full_forward(model):
    ids = torch.randint(3, 512, (2, 9))
    full = model(input_ids=ids, use_cache=False).logits
    prefill = model(input_ids=ids[:, :5], use_cache=True)
    decode = model(input_ids=ids[:, 5:], past_key_values=prefill.past_key_values, use_cache=True)
    assert torch.allclose(full[:, 5:], decode.logits, atol=1e-5)


def test_generate_matches_reference_loops(model):
    ids = torch.randint(3, 512, (2, 6))
    tokens = model.generate(ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(tokens, greedy_no_cache(model, ids, 8))
    assert torch.equal(tokens, generate_with_cache(model, ids, max_new_tokens=8))


def test_left_padding_changes_nothing(model):
    ids = torch.randint(3, 512, (1, 5))
    padded = torch.cat([torch.zeros(1, 2, dtype=torch.long), ids], dim=1)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1]])
    a = model.generate(ids, max_new_tokens=6, do_sample=False)
    b = model.generate(input_ids=padded, attention_mask=mask, max_new_tokens=6, do_sample=False, pad_token_id=0)
    assert torch.equal(a[0], b[0, 2:])


def test_trainer_equals_raw_loop_because_forward_accepts_kwargs(model):
    # **kwargs in forward is what lets Trainer pass num_items_in_batch.
    data = random_token_dataset(32, 512)
    collate = CausalLMCollator(pad_token_id=0)
    cfg = LoopConfig(max_steps=8, warmup_steps=1)
    fresh = BackwardPassForCausalLM(BackwardPassConfig())
    raw = train_raw(clone_model(fresh), data, collate, cfg)
    hf = train_with_trainer(clone_model(fresh), data, collate, cfg)
    assert max(abs(a - b) for a, b in zip(raw["loss"], hf["loss"])) < 1e-5


def test_config_rejects_bad_head_count():
    with pytest.raises(ValueError):
        BackwardPassConfig(hidden_size=65, num_attention_heads=4).validate_architecture()
