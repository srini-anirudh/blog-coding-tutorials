import math

import numpy as np
import pytest

from commcost import alphabeta as ab
from commcost import fabric, ladder
from commcost import overlap as O


def test_fit_recovers_alpha_beta():
    n = np.logspace(1, 8, 20)
    t = 7e-6 + n / 30e9
    a, b = ab.fit_alpha_beta(n, t * (1 + 0.01 * np.sin(n)))
    assert a == pytest.approx(7e-6, rel=0.05)
    assert 1 / b == pytest.approx(30e9, rel=0.05)


def test_n_half_gives_half_bandwidth():
    a, b = 12e-6, 1 / 42e9
    nh = ab.n_half(a, b)
    assert ab.effective_bandwidth(nh, a, b) == pytest.approx(0.5 / b)


def test_tree_ring_crossover_at_post_settings():
    c = ab.crossover_size(ab.t_tree, ab.t_ring, 64, 12e-6, 1 / 45e9)
    assert ab.t_tree(c * 0.99, 64, 12e-6, 1 / 45e9) < ab.t_ring(c * 0.99, 64, 12e-6, 1 / 45e9)
    assert 5 * 2**20 < c < 7 * 2**20  # not ~100 KiB as the post's text says
    assert ab.crossover_size(ab.t_tree, ab.t_ring, 64, 2e-6, 1 / 45e9) < c


def test_bus_bandwidth_convention():
    assert ab.bus_bandwidth(1e9, 1.0, 2) == pytest.approx(1e9)
    assert ab.bus_bandwidth(1e9, 1.0, 8) == pytest.approx(1.75e9)


def test_ladder_numbers():
    assert ladder.pcie_link_bytes_per_s() / 1e9 == pytest.approx(63.0, abs=0.05)
    assert ladder.break_even_intensity(3.35e12) == pytest.approx(295, abs=1)
    assert ladder.full_mesh_links(8) == (28, 7)
    assert ladder.fat_tree_hosts(64) == 65536
    assert ladder.bandwidth_delay_product(50e9, 10e-6) == pytest.approx(500e3)


def test_llama_tensor_model():
    t = O.llama7b_tensors()
    assert len(t) == 291
    assert sum(g.params for g in t) == pytest.approx(6.74e9, rel=0.01)


def test_no_overlap_exposes_whole_allreduce():
    r = O.simulate(O.post_model_tensors(), 0, 64, 12e-6, 1 / 45e9, overlap=False)
    assert r["exposed_s"] == pytest.approx(ab.t_ring(14e9, 64, 12e-6, 1 / 45e9), rel=1e-6)
    assert r["messages"] == 1


def test_bucket_optimum_grows_with_p_not_bandwidth():
    sizes = [b * 1e6 for b in (25, 50, 100, 200, 400, 800, 1600, 3200, 6400)]
    t = O.llama7b_tensors()
    best = lambda p, bw: O.best_bucket(O.sweep(t, sizes, p, 12e-6, 1 / bw))["bucket_bytes"]  # noqa: E731
    assert best(1024, 45e9) > best(64, 45e9)
    assert best(64, 22.5e9) == best(64, 45e9)


def test_ecmp_busiest_link_exceeds_asymptotic_estimate_at_64():
    m = fabric.ecmp_max_load(64, 64, 4000, np.random.default_rng(0))
    assert 3.7 < m.mean() < 4.2
    assert fabric.ln_over_lnln(64) == pytest.approx(math.log(64) / math.log(math.log(64)))


def test_spraying_evens_the_load():
    s = fabric.spray_max_load(64, 64, 1000, 500, np.random.default_rng(0))
    assert s.mean() < 1.15


def test_moe_balanced_router_is_balanced_and_skew_is_not():
    rng = np.random.default_rng
    assert fabric.moe_dispatch(2048, 4, 4, 2, 0.0, rng(0))["max_over_mean"] < 1.05
    assert fabric.moe_dispatch(2048, 4, 4, 2, 1.0, rng(0))["max_over_mean"] > 1.3
    capped = fabric.moe_dispatch(2048, 4, 4, 2, 1.0, rng(0), capacity_factor=1.0)
    assert capped["drop_fraction"] > 0
