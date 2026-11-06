import math

import torch
from torch import nn

from loralab.analysis import intruder_count, magnitude_direction, pearson
from loralab.ledger import LLAMA2_7B, TARGET_SETS, break_even_rank, lora_params, nf4_bits_per_param
from loralab.lora import LoRALinear
from loralab.quant import (NF4_PUBLISHED, NF4Linear, dequantize_blockwise, double_dequantize, double_quantize,
                           nf4_codebook, quantize_blockwise, storage_bits_per_param, uniform_codebook)
from loralab.spectrum import decay_spectrum, eckart_young_error, fit_factored, matrix_with_spectrum, truncate


def test_ledger_matches_the_post():
    g = LLAMA2_7B
    assert round(g.total_params() / 1e9, 2) == 6.74
    m = g.matrices()
    assert round(sum(o * i for k, (o, i) in m.items() if k.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))) / 1e6, 1) == 67.1
    assert round(sum(o * i for k, (o, i) in m.items() if k in ("gate_proj", "up_proj", "down_proj")) / 1e6, 1) == 135.3
    assert break_even_rank(4096, 4096) == 2048
    assert lora_params(g, 8, TARGET_SETS["Q,V only"]) == 32 * 2 * 8 * 8192
    assert abs(nf4_bits_per_param() - 4.127) < 1e-3
    assert round(g.total_params() * nf4_bits_per_param() / 8 / 1e9, 2) == 3.48


def test_nf4_built_from_quantiles_matches_published():
    cb = nf4_codebook()
    assert torch.allclose(cb, NF4_PUBLISHED, atol=1e-6)
    assert (cb == 0).any() and not (uniform_codebook() == 0).any()


def test_codewords_survive_a_round_trip():
    w = NF4_PUBLISHED.repeat(8) * 0.3
    codes, absmax = quantize_blockwise(w, NF4_PUBLISHED, 64)
    assert torch.allclose(dequantize_blockwise(codes, absmax, NF4_PUBLISHED, w.shape), w, atol=1e-7)


def test_blocks_contain_outliers_and_nf4_beats_uniform_on_gaussians():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(256, 256, generator=g)
    w_out = w.clone()
    w_out[0, 0] = 100.0

    def err(x, cb, block):
        c, a = quantize_blockwise(x, cb, block)
        return (dequantize_blockwise(c, a, cb, x.shape) - x).norm() / x.norm()

    assert err(w_out, NF4_PUBLISHED, 64) < 0.5 * err(w_out, NF4_PUBLISHED, None)
    assert err(w, NF4_PUBLISHED, 64) < err(w, uniform_codebook(), 64)


def test_double_quant_bits_and_accuracy():
    assert abs(storage_bits_per_param(4096 * 4096) - 4.127) < 1e-3
    assert storage_bits_per_param(4096 * 4096, double_quant=False) == 4.5
    absmax = torch.rand(1000) + 0.1
    back = double_dequantize(*double_quantize(absmax))
    assert ((back - absmax).abs() / absmax).max() < 0.02


def test_nf4_linear_freezes_base_but_passes_gradients():
    torch.manual_seed(0)
    base = nn.Linear(128, 64)
    q = NF4Linear(base)
    x = torch.randn(4, 128, requires_grad=True)
    y = q(x)
    assert (y - base(x)).norm() / base(x).norm() < 0.15
    assert not any(p.requires_grad for p in q.parameters())
    lora = LoRALinear(q, r=4)
    lora(x).sum().backward()
    assert x.grad is not None and lora.lora_B.grad.abs().max() > 0


def test_eckart_young_and_factored_search():
    M = matrix_with_spectrum(decay_spectrum(24, "fast"), 24, 24)
    for r in (1, 5, 12):
        assert math.isclose((M - truncate(M, r)).norm().item(), eckart_young_error(M, r), rel_tol=1e-8)
    fit, curve = fit_factored(M, 4, steps=1500)
    assert fit < 1.02 * eckart_young_error(M, 4) and curve[-1] < curve[0]


def test_magnitude_direction_and_intruders():
    torch.manual_seed(0)
    w0 = torch.randn(32, 16)
    dm, dd = magnitude_direction(w0, 2 * w0)
    assert dm > 0 and abs(dd) < 1e-6
    U, S, Vh = torch.linalg.svd(w0, full_matrices=False)
    assert intruder_count(w0, w0.clone()) == 0
    new = torch.linalg.qr(torch.randn(32, 32))[0][:, -1]    # some direction, mixed into a top singular slot
    spike = 50 * S[0] * torch.outer(new - (U @ (U.T @ new)), torch.randn(16))
    assert intruder_count(w0, w0 + spike) >= 1
    assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1) < 1e-12
