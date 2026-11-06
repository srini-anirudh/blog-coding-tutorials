import pytest
import torch

from harness_lab.kv_reuse import PrefixKV, agent_contexts, full_forward, run_lane


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return LlamaForCausalLM(cfg).eval()


def test_reused_cache_gives_the_same_logits(tiny_model):
    prefix = list(range(10, 74))
    chunks = [list(range(100 + 16 * i, 116 + 16 * i)) for i in range(5)]
    kv = PrefixKV(tiny_model)
    for ctx in agent_contexts(prefix, chunks) + agent_contexts(prefix, chunks, edit_at=3, edit_pos=8):
        got, _ = kv.forward(ctx)
        assert torch.allclose(got, full_forward(tiny_model, ctx), atol=1e-4)


def test_append_computes_only_the_suffix_and_edit_recomputes(tiny_model):
    prefix = list(range(10, 74))
    chunks = [list(range(100 + 16 * i, 116 + 16 * i)) for i in range(4)]
    steps = run_lane(tiny_model, agent_contexts(prefix, chunks), "reuse")
    assert [s.computed for s in steps] == [80, 16, 16, 16]
    edited = run_lane(tiny_model, agent_contexts(prefix, chunks, edit_at=3, edit_pos=8), "reuse")
    assert edited[2].computed == 64 + 48 - 8
