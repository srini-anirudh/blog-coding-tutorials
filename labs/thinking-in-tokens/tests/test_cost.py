import math

import pytest

from thinktok.cost import (
    DENSE_32B, bits_per_step, count_decode_flops, decode_floor_seconds, gen_flops, kv_bytes_per_token,
    non_embedding_params, per_token_attention_share,
)
from tinyllm import ModelConfig, TinyLLM


def test_post_kv_numbers():
    per_tok = kv_bytes_per_token(64, 8, 128, 2)
    assert per_tok == 256 * 1024
    assert per_tok * 10_000 / 1e9 == pytest.approx(2.62, abs=0.01)


def test_post_latency_floor():
    assert decode_floor_seconds(32e9, 2, 3.3e12) * 1e3 == pytest.approx(19.4, abs=0.1)


def test_attention_share_32b():
    assert 30e9 < non_embedding_params(DENSE_32B) < 32e9
    assert 0.01 < per_token_attention_share(DENSE_32B, 1024) < 0.05  # "a few per cent"
    assert 0.5 < per_token_attention_share(DENSE_32B, 32768) < 1.0  # "roughly comparable"


def test_formula_terms():
    b = gen_flops(1000, 2, 8, prompt=10, k=4, exact=True)
    assert b.weights == 2 * 1000 * 4
    assert b.attention == 4 * 2 * 8 * (10 * 4 + 4 * 5 / 2)


@pytest.mark.parametrize("kv_heads", [4, 2])
def test_flop_counter_matches_formula(kv_heads):
    cfg = ModelConfig(vocab_size=64, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=kv_heads,
                      intermediate_size=128, max_seq_len=128, attn_backend="naive")
    r = count_decode_flops(TinyLLM(cfg), prompt=12, k=20)
    assert r["measured_flops"] == r["formula_exact"]
    assert r["kv_bytes_measured"] == r["kv_bytes_formula"]


def test_bits_per_step():
    assert bits_per_step(151_669) == pytest.approx(17.21, abs=0.01)
    assert bits_per_step(2) == 1 == math.log2(2)
