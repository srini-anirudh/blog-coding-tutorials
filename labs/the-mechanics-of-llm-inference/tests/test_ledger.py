"""The post's paper arithmetic, pinned to the values it prints."""

import pytest

from inferlab import ledger as lg

M8, M7 = lg.MODELS["llama-3-8b"], lg.MODELS["llama-2-7b"]
H100 = lg.ACCELERATORS["h100"]


def test_kv_per_token_table2():
    assert lg.kv_bytes_per_token(M7) == 512 * 1024
    assert lg.kv_bytes_per_token(M8) == 128 * 1024
    assert lg.kv_bytes_per_token(lg.MODELS["llama-3-70b"]) == 320 * 1024
    assert lg.kv_bytes(M7, 1, 131072) / 1e9 == pytest.approx(68.7, abs=0.05)


def test_kv_is_linear_in_batch_and_context():
    assert lg.kv_bytes(M8, 4, 1000) == 4 * lg.kv_bytes(M8, 1, 1000) == 2 * lg.kv_bytes(M8, 2, 1000)


def test_cache_saves_417x():
    without, with_cache = lg.no_cache_vs_cache_flops(8.03e9, 1000, 500)
    assert without / with_cache == pytest.approx(417, abs=1)


def test_gemm_intensity_regimes():
    assert lg.gemm_intensity(8192, 4096, 4096) == pytest.approx(1638, abs=1)
    assert lg.gemm_intensity(1, 4096, 4096) == pytest.approx(1.0, abs=0.01)


def test_decode_wall_and_b_star():
    assert 2 * M8.params / H100.bandwidth * 1e3 == pytest.approx(4.79, abs=0.01)
    assert lg.b_star(H100) == pytest.approx(295, abs=1)
    assert lg.b_star(H100, s_w=1) * 2 == pytest.approx(lg.b_star(H100) * 1, rel=1)  # fp8: halve s_w, double pi
    assert lg.b_cross(M8, 8192) == pytest.approx(15, abs=0.1)


def test_batching_ceiling_table4():
    assert lg.decode_intensity_limit(M8, 2048) == pytest.approx(63.8, abs=0.1)
    assert lg.decode_intensity_limit(M8, 131072) == pytest.approx(4.9, abs=0.05)
    # Infinite context: the limit is 2g / s_c, the group size.
    assert lg.decode_intensity_limit(M8, 10**9) == pytest.approx(M8.group, rel=1e-3)
    # I(B) rises with B and approaches the limit from below.
    vals = [lg.decode_intensity(M8, b, 8192) for b in (1, 10, 100, 10_000)]
    assert vals == sorted(vals) and vals[-1] < lg.decode_intensity_limit(M8, 8192)


def test_speculation_moe_tp_static():
    assert lg.spec_expected_tokens(0.8, 4) == pytest.approx(3.36, abs=0.01)
    assert lg.spec_expected_tokens(1.0, 4) == 5
    assert lg.spec_speedup(0.8, 4, 0.05) == pytest.approx(2.8, abs=0.01)
    assert [round(lg.moe_distinct_experts(8, 2, b), 1) for b in (1, 4, 16)] == [2.0, 5.5, 7.9]
    single = 2 * M8.params / H100.bandwidth
    assert lg.tp_decode_step(single, 8, 32) * 1e3 == pytest.approx(0.98, abs=0.01)
    assert lg.static_batch_utilisation([40, 300, 20, 60, 500, 80, 150, 30]) == pytest.approx(0.295)
