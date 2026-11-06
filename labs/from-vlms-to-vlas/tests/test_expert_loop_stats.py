import math

import pytest
import torch

from vlalab.evalstats import normal_interval, power, simulated_power, trials_needed, wilson_interval
from vlalab.expert import ActionExpertPolicy, block_causal_mask
from vlalab.insulation import N_COLOURS, ToyVLA, make_batch
from vlalab.loop import simulate


def small_policy():
    torch.manual_seed(0)
    return ActionExpertPolicy(act_dim=3, state_dim=2, horizon=5, layers=2, d_vlm=32, d_exp=16, heads=2,
                              head_dim=8, mlp_vlm=64, mlp_exp=32)


def test_block_causal_mask():
    m = block_causal_mask([2, 1, 2])
    assert m[:2, 2:].sum() == 0  # the prefix never sees state or actions
    assert m[2, :3].all() and not m[2, 3:].any()  # state sees prefix and itself
    assert m[3:].all()  # actions see everything


def test_cache_is_exact_and_prefix_ignores_actions():
    m = small_policy().eval()
    prefix, state = torch.randn(2, 7, 32), torch.randn(2, 2)
    a, tau = torch.randn(2, 5, 3), torch.tensor([0.2, 0.7])
    with torch.no_grad():
        cache, _ = m.encode_prefix(prefix)
        v_cached = m.velocity(cache, state, a, tau)
        v_full, px = m.full_forward(prefix, state, a, tau)
        _, px2 = m.full_forward(prefix, state, torch.randn(2, 5, 3), tau)
    assert torch.allclose(v_cached, v_full, atol=1e-5)
    assert torch.equal(px, px2)
    assert m.sample(prefix, state, steps=10).shape == (2, 5, 3)


def test_stop_gradient_keeps_flow_gradients_out_of_the_backbone():
    m = small_policy()
    prefix = torch.randn(2, 7, 32)
    for stop, expect_grad in ((False, True), (True, False)):
        m.zero_grad()
        m.flow_loss(prefix, torch.randn(2, 2), torch.randn(2, 5, 3), stop_grad=stop).backward()
        g = m.layers[0].vlm.qkv.weight.grad
        assert (g is not None and g.abs().sum() > 0) == expect_grad
        assert m.layers[0].exp.qkv.weight.grad.abs().sum() > 0


def test_sync_pauses_and_async_hides_latency():
    sync = simulate(H=10, latency_ms=80, mode="sync")
    asyn = simulate(H=10, latency_ms=80, mode="async")
    assert sync["paused_frac"] > 0.25  # 4 steps of waiting per 10 executed
    assert asyn["paused_frac"] <= 4 / 200 + 1e-9  # only the very first wait
    assert asyn["reaction_ms"] >= 80 and sync["reaction_ms"] >= 80


def test_bigger_chunks_call_less_and_react_later():
    small = simulate(H=5, latency_ms=20, mode="async")
    big = simulate(H=50, latency_ms=20, mode="async")
    assert big["calls_per_s"] < small["calls_per_s"]
    assert big["reaction_ms"] > small["reaction_ms"]


def test_blend_smooths_the_switch():
    a = simulate(H=25, latency_ms=80, mode="async", seed=1)
    b = simulate(H=25, latency_ms=80, mode="blend", seed=1)
    assert b["jump_at_switch"] < a["jump_at_switch"]


def test_post_statistics():
    lo, hi = normal_interval(0.5, 20)
    assert (hi - lo) / 2 == pytest.approx(0.219, abs=0.001)  # "about +-22 points"
    assert trials_needed(0.5, 0.6) == 385
    wl, wh = wilson_interval(0.5, 20)
    assert wl > lo and wh < hi and wl + wh == pytest.approx(1.0)
    assert power(0.5, 0.6, 20) < 0.2 and power(0.5, 0.7, 20) < 0.5  # gaps of 20 points or less are usually missed
    assert simulated_power(0.5, 0.6, 385, reps=4000) == pytest.approx(0.8, abs=0.03)
    assert math.isclose(power(0.5, 0.5 + 1e-9, 50), 0.05, abs_tol=1e-3)


def test_insulation_batches_name_one_object():
    b = make_batch(500, range(5), torch.Generator().manual_seed(0))
    assert ((b["colours"] == b["instr"][:, None]).sum(1) == 1).all()
    assert (b["colours"] < N_COLOURS).all() and (b["instr"] < 5).all()
    assert torch.allclose(b["chunk"][:, -1], b["target"])
    m = ToyVLA(d=32, layers=2)
    assert m.pointing_loss(b).ndim == 0
    ids = {id(p) for p in m.backbone_params()}
    assert all(id(p) not in ids for p in m.expert_params())
    assert len(ids) + len(m.expert_params()) == len(list(m.parameters()))
