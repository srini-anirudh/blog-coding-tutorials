import pytest

from whyslow.roofline import H100_SXM, elementwise_counts, matmul_counts, time_saved, verdict

P, B = H100_SXM["bf16_flops"], H100_SXM["hbm_bytes_per_s"]


def test_three_regimes():
    f, b = matmul_counts(8192, 8192, 8192, 2)
    assert verdict(f, b, f / (0.7 * P), P, B).regime == "compute bound"
    f, b = elementwise_counts(1 << 26, 2, 1, 1, 2)
    assert verdict(f, b, b / (0.9 * B), P, B).regime == "bandwidth bound"
    f, b = matmul_counts(64, 64, 64, 2)
    assert verdict(f, b, 5e-6, P, B).regime == "below both roofs"


def test_ridge_and_ceiling():
    v = verdict(1e9, 1e6, 1e-3, P, B)
    assert v.ridge == pytest.approx(P / B)
    assert v.intensity == 1000 and v.ceiling == P  # past the ridge, the flat roof
    v = verdict(1e9, 1e9, 1.0, P, B)
    assert v.ceiling == pytest.approx(B)  # intensity 1: sloped roof


def test_counts():
    assert matmul_counts(2, 3, 4, 2) == (48.0, 2.0 * (8 + 12 + 6))
    assert elementwise_counts(10, 3, 1, 11, 2) == (110, 80.0)


def test_time_budget_numbers_from_the_post():
    assert time_saved(0.04, 3) == pytest.approx(0.027, abs=5e-4)
    assert time_saved(0.18, 2) == pytest.approx(0.09)
