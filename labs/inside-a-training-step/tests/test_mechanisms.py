"""FLOPs, activation lifetimes, recomputation, precision and the optimizer, on small CPU models."""

from dataclasses import replace

import pytest
import torch

from stepanatomy import flops as F
from stepanatomy import lifetime, models, precision, recompute
from stepanatomy.mixed import MasterWeightAdam, ledger
from stepanatomy.optbench import time_adam

TINY = replace(models.SMALL, layers=3, hidden=64, heads=2, intermediate=172, vocab=128, seq=64, batch=2)


def test_backward_is_twice_forward_for_a_gemm():
    f = F.linear_flops(32, 16, 24, input_grad=True, weight_grad=True)
    assert f["forward"] == 2 * 32 * 16 * 24
    assert f["backward"] == 2 * f["forward"]
    assert F.linear_flops(32, 16, 24, input_grad=True, weight_grad=False)["backward"] == f["forward"]
    assert F.linear_flops(32, 16, 24, input_grad=False, weight_grad=True)["backward"] == f["forward"]


def test_model_forward_flops_match_2nd_plus_attention():
    m = models.build(TINY, "naive")
    f = F.model_flops(m, models.random_batch(TINY))
    n, d = F.matmul_params(m), TINY.tokens
    attn = 4 * TINY.batch * TINY.seq**2 * TINY.hidden * TINY.layers
    assert f["forward"] == 2 * n * d + attn
    assert f["backward"] == 2 * f["forward"]


def test_full_recompute_adds_one_forward_of_the_blocks():
    x = models.random_batch(TINY)
    none = F.model_flops(models.build(TINY, "naive"), x)
    full = F.model_flops(recompute.apply(models.build(TINY, "naive"), "full"), x)
    head = 2 * TINY.tokens * TINY.vocab * TINY.hidden  # the LM head is not checkpointed
    # torch's non-reentrant checkpoint stops recomputing once every saved tensor is rebuilt. Nothing
    # saves the output of a block's last GEMM (down_proj), so that GEMM is never rerun.
    down = 2 * TINY.tokens * TINY.intermediate * TINY.hidden * TINY.layers
    assert full["total"] - none["total"] == none["forward"] - head - down


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_recompute_keeps_gradients(mode):
    x = models.random_batch(TINY)
    a, b = models.build(TINY), recompute.apply(models.build(TINY), mode)
    a(x, labels=x).loss.backward()
    b(x, labels=x).loss.backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_mountain_peaks_at_the_turn_and_drains():
    tl = lifetime.track(models.build(TINY), models.random_batch(TINY))
    assert tl.peak_index == tl.turn
    assert tl.events[-1].live_bytes == 0
    fwd = [e.live_bytes for e in tl.events if e.phase == "forward"]
    assert fwd == sorted(fwd)  # only grows in the forward
    spans = [tl.spans[f"block {i}"] for i in range(TINY.layers)]
    for outer, inner in zip(spans, spans[1:]):  # block i's lifetime contains block i+1's
        assert outer[0] < inner[0] and inner[1] < outer[1]


def test_checkpointing_saves_less():
    x = models.random_batch(TINY)
    saved = {m: lifetime.track(recompute.apply(models.build(TINY), m), x).peak for m in ("none", "selective", "full")}
    assert saved["full"] < saved["selective"] < saved["none"]


def test_selective_recompute_reaches_the_posts_16h():
    tl = lifetime.track(recompute.apply(models.build(TINY), "selective"), models.random_batch(TINY))
    per = lifetime.saved_elems_per_token_per_layer(tl, TINY.tokens, TINY.hidden, 4)
    assert per == pytest.approx(8 + 3 * TINY.intermediate / TINY.hidden, abs=0.6)


def test_naive_attention_saves_quadratic_scores():
    def per_block(seq, attn):
        s = replace(TINY, seq=seq, batch=1)
        return lifetime.track(models.build(s, attn), models.random_batch(s)).saved_by_tag["block 1"]

    assert per_block(128, "sdpa") / per_block(64, "sdpa") == pytest.approx(2, rel=0.05)
    assert per_block(128, "naive") / per_block(64, "naive") > 2.1


def test_master_weight_adam_matches_torch_adamw():
    torch.manual_seed(0)
    w = torch.randn(32, 16)
    p16 = w.clone().bfloat16().requires_grad_()
    p32 = p16.detach().float().clone().requires_grad_()
    ours = MasterWeightAdam([p16], lr=1e-2, weight_decay=0.1)
    ref = torch.optim.AdamW([p32], lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
    for _ in range(5):
        g = torch.randn(32, 16).bfloat16()
        p16.grad, p32.grad = g.clone(), g.float()
        ours.step()
        ref.step()
    torch.testing.assert_close(ours.master[0], p32.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(p16.detach(), p32.detach().bfloat16())


def test_ledger_is_16_bytes_per_param():
    m = models.build(TINY, dtype=torch.bfloat16)
    opt = MasterWeightAdam(m.parameters())
    x = models.random_batch(TINY)
    m(x, labels=x).loss.backward()
    opt.step()
    assert ledger(m, opt)["total"] == 16


def test_precision_failures():
    assert precision.ones_sum_stall(1000) == 256
    d = precision.dot_errors()
    assert d["bf16 accumulate"] > 0.1 > 1e-3 > d["fp32 accumulate"]
    u = precision.update_survival()
    assert u["bf16 weight"] == 1.0 and u["fp32 master"] == pytest.approx(u["exact"], rel=1e-5)
    f = precision.range_flush()
    assert f["fp16"] > 0.2 and f["bf16"] == 0 and f["fp16, loss scaled"] == 0


def test_pure_bf16_adam_stalls_and_master_learns():
    pure = precision.regression_run("pure_bf16", steps=100)
    master = precision.regression_run("master", steps=100)
    assert pure[-1] == pure[0]
    assert master[-1] < 0.2 * master[0]


def test_optimizer_bench_runs():
    for v in ("for-loop", "fused", "master"):
        out = time_adam(v, 2e5, 64, "cpu", iters=2)
        assert out["ms"] > 0 and out["params"] > 1e5
