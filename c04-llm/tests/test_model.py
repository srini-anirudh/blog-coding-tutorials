"""Whole-model tests: shapes, causality, weight tying, parameter count, backends."""

from pathlib import Path

import pytest
import torch

from c04_helpers import random_ids, small_config, small_model
from tinyllm import ModelConfig, TinyLLM, count_parameters, get_config, kv_cache_bytes


def test_forward_shapes():
    model = small_model()
    ids = random_ids(3, 11)
    out = model(ids, labels=ids)
    assert out.logits.shape == (3, 11, 97)
    assert out.loss.shape == ()
    assert out.past_key_values is None


def test_causality_future_tokens_do_not_change_past_logits():
    model = small_model()
    ids = random_ids(1, 12)
    changed = ids.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % 97
    a, b = model(ids).logits, model(changed).logits
    torch.testing.assert_close(a[:, :8], b[:, :8])
    assert not torch.allclose(a[:, 8:], b[:, 8:])


def test_weight_tying_shares_storage():
    model = small_model()
    assert model.lm_head.weight is model.embed_tokens.weight
    untied = small_model(tie_word_embeddings=False)
    assert untied.lm_head.weight is not untied.embed_tokens.weight


@pytest.mark.parametrize(
    "overrides",
    [{}, {"tie_word_embeddings": False}, {"num_kv_heads": 1}, {"num_kv_heads": 4}, {"head_dim": 24},
     {"num_experts": 4, "num_experts_per_tok": 2}],
)
def test_parameter_count_formula_matches_model(overrides):
    config = small_config(**overrides)
    model = TinyLLM(config)
    assert count_parameters(config)["total"] == model.num_parameters()


@pytest.mark.parametrize("name,published", [("smollm2-135m", 134_515_008), ("llama3-8b", 8_030_261_248)])
def test_parameter_count_of_real_models(name, published):
    """Instantiate on the meta device (no memory), count, and compare with the published size."""
    config = get_config(name)
    with torch.device("meta"):
        model = TinyLLM(config)
    assert model.num_parameters() == count_parameters(config)["total"] == published


def test_kv_cache_bytes_formula():
    config = get_config("llama3-8b")
    # 32 layers * 8 kv heads * 128 dims * 2 (K and V) * 2 bytes = 128 KiB per token.
    assert kv_cache_bytes(config, seq_len=1) == 131_072


def test_naive_and_sdpa_models_agree():
    sdpa = small_model(attn_backend="sdpa")
    naive = small_model(attn_backend="naive")
    naive.load_state_dict(sdpa.state_dict())
    ids = random_ids(2, 20)
    torch.testing.assert_close(naive(ids).logits, sdpa(ids).logits, atol=1e-5, rtol=1e-5)


def test_sequence_longer_than_max_raises():
    model = small_model(max_seq_len=8)
    with pytest.raises(ValueError):
        model(random_ids(1, 9))


@pytest.mark.parametrize("name", ["tiny", "smollm2-135m", "llama3-8b"])
def test_config_files_match_presets(name):
    path = Path(__file__).resolve().parents[1] / "configs" / f"{name}.json"
    assert ModelConfig.load(path) == get_config(name)
