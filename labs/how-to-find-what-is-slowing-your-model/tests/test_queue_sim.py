import pytest

from whyslow.queue_sim import simulate, steady_state, steps_until_dry


def test_config_a_loses_16_ms_per_step():
    res = simulate(4, 384, 80, prefetch=2, steps=400, prefill=True)
    s = steady_state(res)
    assert s["mean_wait_ms"] == pytest.approx(16.0)
    assert s["mean_wait_ms"] / (80 + s["mean_wait_ms"]) == pytest.approx(1 / 6)
    assert s["depth_hist"][0] > 0 and s["mean_depth"] < 1


def test_config_a_hides_the_deficit_only_while_the_queue_lasts():
    dry = steps_until_dry(simulate(4, 384, 80, prefetch=2, steps=400, prefill=True))
    deeper = steps_until_dry(simulate(4, 384, 80, prefetch=8, steps=400, prefill=True))
    assert dry is not None and deeper > dry  # depth buys time, never throughput
    s = steady_state(simulate(4, 384, 80, prefetch=8, steps=600, prefill=True), skip=300)
    assert s["mean_wait_ms"] == pytest.approx(16.0)


def test_config_b_never_waits_and_sits_near_the_cap():
    res = simulate(8, 384, 80, prefetch=2, steps=400, prefill=True)
    s = steady_state(res)
    assert steps_until_dry(res) is None and s["mean_wait_ms"] == 0
    assert s["mean_depth"] > 0.7 * res["cap"]


def test_jitter_runs_and_is_reproducible():
    a = simulate(8, 384, 80, steps=200, jitter=0.3, seed=1)
    b = simulate(8, 384, 80, steps=200, jitter=0.3, seed=1)
    assert a == b
