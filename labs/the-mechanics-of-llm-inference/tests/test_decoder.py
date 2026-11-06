"""The static-cache decoder: cache paths agree with full recomputation, and it matches tinyllm and HF Llama."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from inferlab.decoder import SHAPES, Decoder, DecoderConfig, greedy_generate

C4 = Path(__file__).resolve().parents[3] / "c04-llm"


def tiny(seed=0, max_len=64, **kw):
    cfg = SHAPES["tiny"] if not kw else DecoderConfig(**{**SHAPES["tiny"].__dict__, **kw})
    m = Decoder.random(cfg, max_len, seed=seed)
    m.allocate_cache(2)
    return m


def test_decode_steps_match_full_forward():
    m = tiny()
    ids = torch.randint(0, 256, (2, 20))
    full = m.forward(ids, 0, last_only=False)
    m.forward(ids[:, :12], 0)
    steps = [m.forward(ids[:, t : t + 1], t)[:, 0] for t in range(12, 20)]
    torch.testing.assert_close(torch.stack(steps, 1), full[:, 12:], rtol=1e-5, atol=1e-5)


def test_multi_row_verify_matches_full_forward():
    """The speculative verify pass: several new rows at once after a cached prefix."""
    m = tiny()
    ids = torch.randint(0, 256, (2, 20))
    full = m.forward(ids, 0, last_only=False)
    m.forward(ids[:, :15], 0)
    torch.testing.assert_close(m.forward(ids[:, 15:], 15, last_only=False), full[:, 15:], rtol=1e-5, atol=1e-5)


def test_greedy_with_and_without_cache_agree():
    m = tiny(seed=3)
    m.allocate_cache(1)
    p = torch.randint(0, 256, (1, 7))
    assert torch.equal(greedy_generate(m, p, 15, True), greedy_generate(m, p, 15, False))


def test_cache_bytes_follow_the_formula():
    m = tiny()
    c = m.config
    assert m.allocate_cache(3, 40) == 2 * 3 * 40 * c.layers * c.n_kv * c.head_dim * 4  # fp32: 4 bytes


def test_decode_read_excludes_embedding_table_only_when_untied():
    m = tiny()
    emb = m.embed_tokens.weight.numel() * 4
    assert m.weight_bytes() - m.weight_bytes(read_per_decode_step=True) == emb
    t = tiny(tied=True)
    assert t.lm_head.weight is t.embed_tokens.weight
    assert t.weight_bytes() == t.weight_bytes(read_per_decode_step=True)


def test_overflow_raises():
    m = tiny(max_len=16)
    with pytest.raises(ValueError):
        m.forward(torch.zeros(1, 17, dtype=torch.long), 0)


@pytest.mark.skipif(not (C4 / "tinyllm").exists(), reason="c04-llm not present")
def test_matches_c4_tinyllm():
    sys.path.insert(0, str(C4))
    from tinyllm import ModelConfig, TinyLLM

    torch.manual_seed(0)
    ref = TinyLLM(ModelConfig(vocab_size=256, hidden_size=128, num_layers=4, num_heads=4, num_kv_heads=2,
                              intermediate_size=352, max_seq_len=64)).eval()
    m = Decoder(SHAPES["tiny"], 64)
    m.load_state_dict({k: v for k, v in ref.state_dict().items() if k in m.state_dict()})
    m._build_rope("cpu")
    m.allocate_cache(1)
    ids = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        torch.testing.assert_close(m.forward(ids, 0, last_only=False), ref(ids).logits, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers not installed")
def test_matches_hf_llama_random_weights():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    hf = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                                      num_key_value_heads=2, intermediate_size=352, rope_theta=10_000.0,
                                      max_position_embeddings=64, tie_word_embeddings=False, rms_norm_eps=1e-5)).eval()
    m = Decoder(SHAPES["tiny"], 64)
    m.load_state_dict({k.removeprefix("model."): v for k, v in hf.state_dict().items()})
    m._build_rope("cpu")
    m.allocate_cache(1)
    ids = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        torch.testing.assert_close(m.forward(ids, 0, last_only=False), hf(ids).logits, rtol=1e-4, atol=1e-4)
