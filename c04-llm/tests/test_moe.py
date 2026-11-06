"""Mixture of experts: routing, capacity, and the load-balancing loss."""

import torch

from c04_helpers import random_ids, small_model
from tinyllm.mlp import SwiGLU
from tinyllm.moe import MoE, load_balancing_loss


def test_moe_shapes_and_counts():
    torch.manual_seed(0)
    moe = MoE(16, 32, num_experts=4, top_k=2)
    out, aux = moe(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16) and aux.shape == ()
    assert moe.last_tokens_per_expert.sum().item() == 2 * 5 * 2  # N tokens * k assignments
    assert moe.last_dropped == 0


def test_top1_moe_with_one_expert_is_that_expert():
    """With E = k = 1 the router weight is always 1, so MoE reduces to a plain SwiGLU."""
    torch.manual_seed(0)
    moe = MoE(16, 32, num_experts=1, top_k=1)
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(moe(x)[0], moe.experts[0](x))
    assert isinstance(moe.experts[0], SwiGLU)


def test_moe_matches_dense_reference():
    """Compute every expert on every token, then mix with the renormalised top-k weights."""
    torch.manual_seed(0)
    moe = MoE(16, 32, num_experts=4, top_k=2)
    x = torch.randn(1, 7, 16)
    flat = x.view(-1, 16)
    probs = torch.softmax(moe.router(flat), dim=-1)
    top_p, top_idx = probs.topk(2, dim=-1)
    top_p = top_p / top_p.sum(-1, keepdim=True)
    all_out = torch.stack([e(flat) for e in moe.experts], dim=1)  # (N, E, D)
    ref = (all_out.gather(1, top_idx[..., None].expand(-1, -1, 16)) * top_p[..., None]).sum(1)
    torch.testing.assert_close(moe(x)[0].view(-1, 16), ref, atol=1e-6, rtol=1e-5)


def test_capacity_drops_overflow_tokens():
    torch.manual_seed(0)
    moe = MoE(16, 32, num_experts=4, top_k=1, capacity_factor=1.0)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] = 10.0  # push every token to expert 0
    x = torch.ones(1, 8, 16)
    moe(x)
    assert moe.last_tokens_per_expert.tolist() == [2, 0, 0, 0]  # capacity = ceil(1.0 * 8 * 1 / 4) = 2
    assert moe.last_dropped == 6


def test_load_balancing_loss_is_one_when_uniform_and_larger_when_collapsed():
    n, e = 8, 4
    uniform = torch.full((n, e), 1 / e)
    spread_idx = torch.arange(n).remainder(e)[:, None]
    torch.testing.assert_close(load_balancing_loss(uniform, spread_idx, e), torch.tensor(1.0))
    collapsed = torch.tensor([[0.97, 0.01, 0.01, 0.01]]).expand(n, e)
    assert load_balancing_loss(collapsed, torch.zeros(n, 1, dtype=torch.long), e) > 3.5


def test_moe_model_trains_router():
    model = small_model(num_experts=4, num_experts_per_tok=2).train()
    ids = random_ids(2, 16)
    out = model(ids, labels=ids)
    assert out.aux_loss is not None and out.aux_loss.item() >= 1.0 * model.config.num_layers - 1e-3
    out.loss.backward()
    assert model.layers[0].mlp.router.weight.grad.abs().sum() > 0
