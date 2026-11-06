"""Heads, linear recall, MLA, IO counts and the budget arithmetic."""

import pytest
import torch

from attnmoves import budgets, io
from attnmoves.common import use_c03
from attnmoves.heads import MultiHeadRouter, train
from attnmoves.latent import absorption_check, best_fixed_matrix_error, cache_per_token, decoupled_check
from attnmoves.recall import accuracy, linear_state, make_pairs, recall_linear, recall_softmax


def test_heads_partition_parameters():
    counts = {sum(p.numel() for p in MultiHeadRouter(12, 12, heads=h).parameters()) for h in (1, 2, 4, 8)}
    assert len(counts) == 1


def test_two_heads_do_two_lookups_one_head_cannot():
    two = train(2, steps=600)[1]
    one = train(1, steps=600)[1]
    assert two["acc(i-1)"] > 0.95 and two["acc(i-3)"] > 0.95
    assert min(one["acc(i-1)"], one["acc(i-3)"]) < 0.3


def test_linear_state_is_the_matrix_product():
    k, v = make_pairs(20, 8)
    s, u = linear_state(k, v)
    assert torch.allclose(s, k.T @ v, atol=1e-6)
    assert torch.allclose(u, k.sum(0), atol=1e-6)


def test_recall_softmax_exact_linear_fails_past_capacity():
    k, v = make_pairs(1024, 64)
    assert accuracy(recall_softmax(k, v), v) == 1.0
    assert accuracy(recall_linear(k, v), v) < 0.3
    k, v = make_pairs(16, 64)
    assert accuracy(recall_linear(k, v), v) == 1.0


def test_absorption_rope_breaks_it_decoupled_fixes_it():
    a = absorption_check()
    assert a["no RoPE: max |explicit - absorbed|"] < 1e-10
    assert a["RoPE on key: max |true - naive absorbed|"] > 1.0
    assert best_fixed_matrix_error() > 0.3
    d = decoupled_check()
    assert d["decoupled RoPE: max |explicit - absorbed|"] < 1e-10
    assert d["decoupled RoPE, absorbed and cached: max |cached - full|"] < 1e-10


@pytest.mark.parametrize("n,block", [(64, 16), (100, 32), (256, 64)])
def test_io_counts_match_formulas_and_outputs_match(n, block):
    q, k, v = (torch.randn(n, 16, dtype=torch.float64) for _ in range(3))
    m1, m2 = io.SlowMemory(), io.SlowMemory()
    ref = io.textbook_attention(q, k, v, m1)
    out, lse = io.tiled_attention(q, k, v, m2, block=block)
    assert m1.total == io.textbook_formula(n, 16)
    assert m2.total == io.tiled_formula(n, 16, block)
    assert torch.allclose(out, ref, atol=1e-12)
    assert torch.allclose(lse, torch.logsumexp(q @ k.T / 4.0, -1), atol=1e-12)


def test_our_tiled_kernel_matches_c03():
    use_c03()
    from attnzoo.attention_tiled import tiled_attention as c03_tiled

    q, k, v = (torch.randn(1, 1, 96, 16, dtype=torch.float64) for _ in range(3))
    ours = io.tiled_attention(q[0, 0], k[0, 0], v[0, 0], io.SlowMemory(), block=32)[0]
    assert torch.allclose(ours, c03_tiled(q, k, v, block_q=32, block_k=32)[0, 0], atol=1e-5)  # C3 tiles in fp32


def test_budget_numbers_quoted_in_the_post():
    m = budgets.LLAMA2_7B
    assert budgets.kv_cache_bytes(m["layers"], 1, 8192, 32, m["dh"]) == 4 * budgets.GIB
    d = budgets.DEEPSEEK_V2
    assert cache_per_token(d["heads"], d["dh"], d_c=d["d_c"], d_r=d["d_r"]) == 576
    assert cache_per_token(d["heads"], d["dh"]) == 32768
    assert budgets.vit_tokens(224, 14) == 256 and budgets.vit_tokens(1024, 14) == 5329
    assert budgets.decode_intensity(8192, 32, 32, 128) == 1.0
    assert budgets.receptive_field(4096, 32) == 131072


@pytest.mark.gpu
def test_flash_matches_math_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q, k, v = (torch.randn(1, 4, 512, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    outs = []
    for b in (SDPBackend.MATH, SDPBackend.FLASH_ATTENTION):
        with sdpa_kernel(b):
            outs.append(torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).float())
    assert (outs[0] - outs[1]).abs().max().item() < 2e-2
