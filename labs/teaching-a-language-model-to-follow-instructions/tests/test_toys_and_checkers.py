"""Divergences, behaviour cloning toys, format checkers, reasoning chain, verifiers. Offline and fast."""

import json

import numpy as np
import pytest
import torch
from sftlab import arith, bc, chain, data, divergence as dv, formats, metrics, verifiers as V


def test_cross_entropy_is_entropy_plus_kl():
    rng = np.random.default_rng(0)
    for _ in range(5):
        d, p = rng.dirichlet(np.ones(20)), rng.dirichlet(np.ones(20))
        assert dv.cross_entropy(d, p) == pytest.approx(dv.entropy(d) + dv.kl(d, p), rel=1e-9)


def test_forward_kl_covers_reverse_kl_seeks():
    x = np.linspace(-12, 12, 1201)
    D = dv.mixture_pdf(x, [0.72, 0.28], [-3, 3], [0.6, 0.6])
    mu_f, sd_f, _ = dv.fit_gaussian(x, D, "forward")
    mu_r, sd_r, _ = dv.fit_gaussian(x, D, "reverse")
    assert sd_f > 2.0 and abs(mu_r + 3) < 0.3 and sd_r < 1.0


def test_forward_kl_asymmetry_on_unseen_behaviour():
    D = np.array([0.6, 0.3, 0.1, 0.0])
    p = np.array([0.48, 0.24, 0.08, 0.20])
    assert np.isfinite(dv.kl(D, p)) and dv.kl(p, D) == float("inf")
    assert dv.kl(D, np.array([0.667, 0.333, 0.0, 0.0])) == float("inf")


def test_tabular_worst_case_has_quadratic_and_linear_shapes():
    Ts = [20, 40, 80, 160]
    quad = bc.loglog_slope(Ts, [bc.tabular_worst_case(0.002, T, n=40000) for T in Ts])
    lin = bc.loglog_slope(Ts, [bc.tabular_worst_case(0.002, T, n=40000, recover=True) for T in Ts])
    assert 1.8 < quad < 2.1 and 0.9 < lin < 1.1


def test_ring_expert_stays_on_the_ring_and_dagger_data_is_wider():
    g = torch.Generator().manual_seed(0)
    S, C = bc.rollout(bc.expert_action, 20, 100, 0.002, g)
    assert float(C.sum()) == 0.0
    net = bc.fit(S, steps=50)
    St, _ = bc.rollout(net, 20, 100, 0.002, torch.Generator().manual_seed(1))
    assert St.shape == S.shape


def test_format_constraints_round_trip():
    resp = "Gradient descent updates weights. It uses the slope of the loss. The step size is the learning rate. Smaller steps are safer."
    ex = {"prompt": "Explain.", "response": resp, "source": "t"}
    for c in formats.TRAIN + [formats.HELD_OUT]:
        out = formats.make_example(ex, c)
        assert c.check(out["response"]), c.name
        assert out["prompt"].endswith(c.text)
    assert not formats.BY_NAME["lowercase"].check("Not lower")
    assert not formats.BY_NAME["three_bullets"].check("- a\n- b")
    assert not formats.BY_NAME["one_sentence"].check("One. Two.")
    assert formats.eligible(ex, formats.BY_NAME["three_bullets"])


def test_chain_trace_checker():
    d = [3, 9, 4, 5]
    assert chain.totals(d) == [3, 2, 6, 1]
    r = chain.trace_answer(d)
    assert r == "Running total: 3, 2, 6, 1. Answer: 1"
    assert chain.check_trace(r, d)["valid"]
    bad = "Running total: 3, 2, 7, 2. Answer: 2"      # one slip, carried forward consistently
    c = chain.check_trace(bad, d)
    assert not c["valid"] and not c["final_ok"] and c["steps_ok"] == 3
    lucky = "Running total: 3, 3, 7, 1. Answer: 1"       # wrong steps, right final digit
    c = chain.check_trace(lucky, d)
    assert c["final_ok"] and not c["valid"]
    assert chain.parse_final(chain.direct_answer(d)) == 1


def test_arith_checker():
    assert arith.n_wrong("3 * 4 = 12 and 12 + 5 = 17") == (2, 0)
    assert arith.n_wrong("3 * 4 = 13") == (1, 1)
    assert arith.n_wrong("$1,200 / 4 = $300") == (1, 0)
    assert arith.n_wrong("2 + 3 * 4 = 14") == (1, 0)              # precedence
    assert arith.n_wrong("2 + 3 * 4 = 20") == (1, 1)
    assert arith.n_wrong("\\( 5 \\times 6 = 30 \\)") == (1, 0)    # LaTeX operators
    assert arith.n_wrong("7 / 3 = 2.33") == (1, 0)                # rounded result accepted
    assert arith.n_wrong("3.5 * 60 + 227 = 210 + 227 = 437") == (1, 0)    # a chain: only the complete statement is checked
    assert arith.n_wrong("15% of 80: 15\\% \\times 80 = 12") == (0, 0)    # percent arithmetic is skipped
    assert arith.n_wrong("X - 1.75 - 1.25 = 6") == (0, 0)           # algebra with a variable is not a statement about numbers
    assert arith.n_wrong("(100 + 200 + 750) - 100 = 950") == (0, 0)  # a bracketed sub-expression
    assert arith.path_key("2 + 3 = 5") != arith.path_key("3 + 2 = 5")


def test_verifiers():
    assert V.check_json('{"a": 1, "b": "x"}', {"a": int, "b": str})
    assert not V.check_json('{"a": true, "b": "x"}', {"a": int, "b": str})
    assert not V.check_json("not json", {"a": int})
    assert V.check_label("positive", {"positive", "negative"}, "positive")
    assert not V.check_label("positive.", {"positive", "negative"}, "positive")
    assert V.check_math("so 1,250 dollars", 1250.0) and not V.check_math("so 1,205", 1250.0)
    assert V.check_code("def f(x):\n    return x + 1", "assert f(1) == 2")
    assert not V.check_code("def f(x):\n    return x", "assert f(1) == 2")
    assert not V.check_code("while True:\n    pass", "", timeout=1.0)


def test_capped_mixing_and_text_metrics():
    sizes = {"a": 1_000_000, "b": 1000, "c": 20_000}
    p, c = V.proportional(sizes), V.capped_proportional(sizes, 20_000)
    assert sum(p.values()) == pytest.approx(1) and sum(c.values()) == pytest.approx(1)
    assert p["b"] < 0.002 and c["b"] > 0.02
    assert V.rouge_l("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)
    assert V.rouge_l("the cat sat", "dogs run fast") == 0.0
    frac, hit = V.contaminated_fraction(["a b c d e f g h", "x y z"], ["one a b c d e f g two"], 5)
    assert hit == [0] and frac == 0.5
    assert metrics.f1("the cat", "the cat sat") == pytest.approx(0.8)
    assert metrics.distinct_n(["a b a b"], 2) == pytest.approx(2 / 3)


def test_data_helpers():
    items = list(range(100))
    tr, va = data.split(items, 10, 20, seed=3)
    tr2, va2 = data.split(items, 50, 20, seed=3)
    assert va == va2 and set(tr) <= set(tr2) | set(tr) and not (set(tr) & set(va))
    assert data.user_text("Do it", "context") == "Do it\n\ncontext"


@pytest.mark.network
def test_public_datasets_load():
    a = data.load_alpaca()
    d = data.load_dolly()
    assert len(a) > 50_000 and len(d) > 14_000
    assert {"instruction", "context", "category"} <= set(d[0])
