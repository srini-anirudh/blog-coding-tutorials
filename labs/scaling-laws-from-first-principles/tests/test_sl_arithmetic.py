"""The closed-form pieces: FLOP counting, allocation, lifetime cost, repetition."""

import math

import numpy as np
import pytest

from scalelab import flops
from scalelab.allocation import HOFFMANN, LossLaw, effective_tokens, lifetime_optimal, optimal, optimal_by_search, plan
from scalelab.allocation import shortfalls
from scalelab.ladder import LADDER, build


@pytest.mark.parametrize("shape", [LADDER[0], LADDER[2]])
def test_six_flops_per_weight_per_token(shape):
    g = flops.measure(build(shape), batch=1, seq=shape.seq, vocab=shape.vocab)
    p = shape.linear_params + shape.vocab * shape.d  # tied head counts once, as a matmul
    assert g["fwd_matmul"] == pytest.approx(2 * p)
    assert g["bwd_matmul"] == pytest.approx(4 * p)
    assert g["fwd_attn"] == pytest.approx(4 * shape.layers * shape.d * shape.seq)  # unmasked QK^T and AV


def test_ladder_counts_match_the_model():
    for s in LADDER[:4]:
        assert sum(p.numel() for p in build(s).parameters()) == s.total


def test_conventions_fan_apart_as_width_shrinks():
    ratios = [flops.conventions(d, 32, 100_352, 8_192)["kaplan"] / flops.conventions(d, 32, 100_352, 8_192)["reference"]
              for d in (4096, 1024, 256)]
    assert ratios[0] > ratios[1] > ratios[2]
    c = flops.conventions(4096, 32, 100_352, 8_192)
    assert c["kaplan"] == 72 * 32 * 4096**2


@pytest.mark.parametrize("C", [1e19, 1e23, 3.8e25])
def test_closed_form_matches_search_and_invariant(C):
    n, d = optimal(HOFFMANN, C)
    ns, _ = optimal_by_search(HOFFMANN, C)
    assert n == pytest.approx(ns, rel=1e-3)
    assert 6 * n * d == pytest.approx(C)
    tn, td = shortfalls(HOFFMANN, n, d)
    assert HOFFMANN.alpha * tn == pytest.approx(HOFFMANN.beta * td)


def test_exponents_sum_to_one_and_gopher_numbers():
    assert HOFFMANN.a + HOFFMANN.b == pytest.approx(1)
    n, d = optimal(HOFFMANN, 5.76e23)
    assert n / 1e9 == pytest.approx(32.2, abs=0.1)
    assert d / n == pytest.approx(93, abs=1)


def test_equal_exponents_give_a_constant_ratio():
    law = LossLaw(1.7, 400, 400, 0.3, 0.3)
    r = [optimal(law, C)[1] / optimal(law, C)[0] for C in (1e18, 1e24)]
    assert r[0] == pytest.approx(r[1])


def test_inference_demand_shrinks_the_model():
    ns = [lifetime_optimal(HOFFMANN, 1.9, t)[0] for t in (0, 1e12, 1e13, 1e14)]
    assert all(a > b for a, b in zip(ns, ns[1:]))
    assert lifetime_optimal(HOFFMANN, 1.5, 0) is None  # below the floor E = 1.69


def test_effective_tokens():
    assert effective_tokens(100, 1) == 100
    vals = [effective_tokens(100, e) for e in (1, 2, 4, 16, 1000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    assert vals[-1] <= 100 * (1 + 15.4) + 1e-9  # saturates at the (1 + R*) U ceiling
    assert effective_tokens(100, 4) / 400 > 0.9


def test_planner_budget():
    p = plan(1024, 720, 989e12, 0.42, 6e12)
    assert p.budget == pytest.approx(1024 * 720 * 3600 * 989e12 * 0.42)
    assert math.isclose(6 * p.n * p.d, p.budget, rel_tol=1e-9)
    assert np.isfinite(p.epochs)
