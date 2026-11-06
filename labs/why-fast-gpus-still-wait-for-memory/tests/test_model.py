"""The formulas reproduce the numbers the post derives."""

import pytest

from rooflab import model as M
from rooflab.hardware import ACCELS, H100, LINKS

S = 2


def test_ridge_points_match_table_1():
    expect = {"A100 SXM 80GB": 153, "H100 SXM5": 295, "H200 SXM": 206, "B200 (spec)": 281, "MI300X": 247}
    for name, ridge in expect.items():
        assert ACCELS[name].ridge("bf16") == pytest.approx(ridge, abs=1)
    assert H100.ridge("fp8") == pytest.approx(591, abs=1)
    assert H100.ridge("fp32") == pytest.approx(20, abs=0.5)


def test_add_vs_matmul():
    assert M.elementwise_intensity(1, 3, S) == pytest.approx(1 / 6)
    assert M.gemm_intensity(4096, 4096, 4096, S) == pytest.approx(4096 / 3)


def test_roofline_and_serial_model():
    ridge = H100.ridge()
    assert M.attainable(ridge, H100.bf16, H100.hbm) == pytest.approx(H100.bf16)
    assert M.attainable(1.0, H100.bf16, H100.hbm) == pytest.approx(H100.hbm)
    # No overlap costs exactly 2x at the ridge and converges to the roof far from it.
    assert M.attainable_serial(ridge, H100.bf16, H100.hbm) == pytest.approx(H100.bf16 / 2)
    assert M.attainable_serial(1e6, H100.bf16, H100.hbm) / H100.bf16 > 0.99
    assert M.fraction_of_peak(2, ridge) == pytest.approx(0.0068, abs=1e-4)
    assert M.roofline_time(10, 100, 1, 1) == 100


def test_gemm_tiling():
    assert 3 * S * H100.ridge() / 2 == pytest.approx(885, abs=2)
    assert M.gemm_intensity_tiled_limit(S, 128, 128) == 64
    assert M.gemm_intensity_tiled_limit(S, 256, 256) == 128
    n = 8192
    i = M.gemm_flops(n, n, n) / M.gemm_bytes_tiled(n, n, n, S, 128, 128)
    assert 60 < i < 64  # approaches T/s from below because of the output write


def test_table_2_projection():
    for tokens, i in [(32768, 3641), (512, 455), (64, 63), (1, 1.0)]:
        assert M.projection_intensity(tokens, 8192, S) == pytest.approx(i, rel=0.01)


def test_decode_economics():
    assert M.decode_step_time(70e9, S, H100.hbm) == pytest.approx(41.8e-3, rel=0.01)
    assert M.batch_to_ridge(H100.ridge(), S) == pytest.approx(295, abs=1)
    assert M.kv_bytes_per_token(S, 128, 8, 80) == 320 * 1024
    kv_seq = M.kv_bytes_per_token(S, 128, 8, 80) * 8192
    assert M.kv_crossover_batch(70e9, S, kv_seq) == pytest.approx(53, abs=1.5)


def test_attention_intensities():
    n, d = 1 << 20, 128
    assert M.attention_flops(n, d) / M.attention_bytes_naive(n, d, S) == pytest.approx(64, rel=0.001)
    assert M.attention_flops(2048, d) / M.attention_bytes_compulsory(2048, d, S) == 1024
    assert M.attention_flops(n, d) / M.attention_bytes_flash_tiled(n, d, S, 128) == pytest.approx(128, rel=0.001)
    assert M.decode_attention_intensity(S) == 1
    assert M.decode_attention_intensity(S, 8) == 8


def test_optimizer_bytes_match_post():
    assert M.optimizer_bytes("sgd", bf16_copy=True).total == 14
    assert M.optimizer_bytes("sgd_momentum", bf16_copy=True).total == 22
    assert M.optimizer_bytes("adam", bf16_copy=True).total == M.ADAM_POST.total == 30
    assert M.ADAM_POST.flops / M.ADAM_POST.total == pytest.approx(0.4)
    assert 30 * 7e9 / H100.hbm == pytest.approx(62e-3, rel=0.02)


def test_little_and_links():
    assert M.little_bytes_in_flight(H100.hbm, 600e-9) == pytest.approx(2.0e6, rel=0.01)
    assert H100.bf16 / LINKS["NVLink 4"] == pytest.approx(2198, abs=1)
    assert H100.bf16 / LINKS["400 Gb/s fabric"] == pytest.approx(19780, abs=10)


def test_chain_tensor_passes():
    assert M.chain_tensor_passes([True] * 4) == 12  # 3 + 2 + 2 + 2 + 3
    assert M.chain_tensor_passes([False] * 4) == 5  # read x, r, g; write h, y
    assert M.chain_tensor_passes([True, False, False, False]) == 3 + 3
    # Removing a boundary saves one write and one read, except after the residual add: h is the residual
    # stream and is written anyway, so fusing past it saves only the read.
    for k, saved in enumerate([1, 2, 2, 2]):
        b = [True] * 4
        b[k] = False
        assert M.chain_tensor_passes(b) == 12 - saved
