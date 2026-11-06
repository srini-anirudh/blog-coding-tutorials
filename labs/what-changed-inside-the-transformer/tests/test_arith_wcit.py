"""The post's numbers, recomputed. Each assert quotes the sentence it checks."""

import math

import pytest

from attnzoo.kv_cache import kv_cache_bytes as c03_kv_bytes
from attnzoo.kv_cache import mla_cache_bytes as c03_mla_bytes
from organs.arith import (GIB, KIB, MIB, PRESETS, calc_model, decode_intensity, kv_bytes, logits_bytes,
                          mha_equivalent_bytes_per_token, mla_bytes_per_token, ridge_point, routing_configurations,
                          swa_bytes, swiglu_parity_width)


def test_llama3_mha_cache():
    # "With full multi-head attention that is 512 KiB per token, so a single 128k-token sequence needs 64 GiB"
    assert kv_bytes(32, 32, 128, 1) == 512 * KIB
    assert kv_bytes(32, 32, 128, 131072) == 64 * GIB


def test_matches_c03_formulas():
    assert kv_bytes(7, 3, 64, 100, batch=2, bytes_per=4) == c03_kv_bytes(7, 2, 100, 3, 64, 4)
    assert mla_bytes_per_token(61, 512, 64) == c03_mla_bytes(61, 1, 1, 512, 64, 2)


@pytest.mark.parametrize("dh,s", [(64, 10), (128, 4096), (256, 131072)])
def test_intensity_is_head_ratio(dh, s):
    # "Independent of s, d_h, and batch. It is purely the query-to-KV head ratio."
    for nq, nkv in [(32, 32), (32, 8), (64, 4), (32, 1)]:
        assert decode_intensity(nq, nkv, dh, s) == pytest.approx(nq / nkv)


def test_ridge_point():
    # "its ridge point is near 295 FLOPs per byte"
    assert round(ridge_point(989e12, 3.35e12)) == 295


def test_mla_numbers():
    # "61 x (512 + 64) x 2 = 70,272 B/token = 68.6 KiB vs 4.77 MiB ... A 71x reduction. 8.6 GiB instead of 610 GiB."
    mla, mha = mla_bytes_per_token(61, 512, 64), mha_equivalent_bytes_per_token(61, 128, 192, 128)
    assert mla == 70272
    assert round(mla / KIB, 1) == 68.6
    assert round(mha / MIB, 2) == 4.77
    assert round(mha / mla) == 71
    assert round(mla * 131072 / GIB, 1) == 8.6
    assert round(mha * 131072 / GIB) == 610


def test_olmo3_sliding_window():
    # "full = 16 GiB ... 3:1 SWA = 4.75 GiB. A 3.4x reduction"
    per = 2 * 8 * 128 * 2
    assert per == 4096
    full, mixed = swa_bytes(64, 0, 4096, 65536, per), swa_bytes(16, 48, 4096, 65536, per)
    assert full == 16 * GIB
    assert mixed == int(4.75 * GIB)
    assert round(full / mixed, 1) == 3.4


def test_swiglu_parity():
    # "d = 4096, 8/3 d = 10,923, shipped as 11,008 - the next multiple of 256"
    exact, rounded = swiglu_parity_width(4096)
    assert round(exact) == 10923 and rounded == 11008
    assert 14336 / 4096 == 3.5


def test_routing_combinatorics():
    # "C(8,2) = 28 versus C(256,8) ~ 4.1 x 10^14"
    assert routing_configurations(8, 2) == 28
    assert f"{routing_configurations(256, 8):.1e}" == "4.1e+14"


def test_logits_tensor():
    # "8 x 8192 x 151,936 x 4 bytes ~ 37 GiB"
    assert round(logits_bytes(8, 8192, 151936) / GIB) == 37


def test_calculator_presets_match_reported_sizes():
    reported = {"gpt2xl": 1.56e9, "llama2": 6.74e9, "llama3": 8.03e9, "olmo3": 32.2e9, "qwen3": 235e9,
                "gptoss": 117e9, "dsv3": 671e9}
    for key, n in reported.items():
        assert calc_model(PRESETS[key])["total"] == pytest.approx(n, rel=0.01), key
    # Active parameters for the MoE table: 22B, 5.1B, 37B.
    assert calc_model(PRESETS["qwen3"])["active"] == pytest.approx(22e9, rel=0.03)
    assert calc_model(PRESETS["gptoss"])["active"] == pytest.approx(5.1e9, rel=0.02)
    assert calc_model(PRESETS["dsv3"])["active"] == pytest.approx(37e9, rel=0.02)


def test_calculator_kv_is_the_formula():
    m = calc_model(PRESETS["llama3"])
    assert m["kv_per_tok"] == 128 * KIB
    assert m["flops"] == 2 * m["active"] + 4 * 32 * 32 * 128 * 8192
    assert math.isclose(m["kv_at_s"], 1 * GIB)


@pytest.mark.network
def test_calculator_against_hub_config():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "11_spec_calculator.py"
    spec = importlib.util.spec_from_file_location("spec_calc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for key in ("gpt2xl", "qwen3"):
        exact = mod.exact_count(mod.HUB_IDS[key])
        assert calc_model(PRESETS[key])["total"] == pytest.approx(exact, rel=0.005)
