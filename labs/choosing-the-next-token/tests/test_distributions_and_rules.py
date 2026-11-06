"""Softmax, entropy, temperature and the six truncation rules, checked against direct definitions."""

import math

import pytest
import torch

from nexttoken import dists, truncation as tr


def test_softmax_is_shift_invariant():
    z = torch.randn(3, 50)
    assert torch.allclose(dists.softmax(z), dists.softmax(z + 123.4), atol=1e-6)


def test_softmax_is_max_entropy_at_fixed_expected_score():
    # Moving mass in any direction that keeps sum(p) and E_p[z] fixed must lower the entropy.
    torch.manual_seed(0)
    z = torch.randn(8)
    p = dists.softmax(z)
    A = torch.stack([torch.ones(8), z])
    for _ in range(50):
        v = torch.randn(8)
        v = v - A.T @ torch.linalg.lstsq(A.T, v).solution  # project onto the null space of the constraints
        q = p + 1e-3 * v / v.norm()
        assert dists.entropy(q) < dists.entropy(p)


def test_dH_dT_matches_autograd():
    z = torch.randn(30) * 3
    for T in [0.3, 1.0, 2.5]:
        t = torch.tensor(T, requires_grad=True)
        p = torch.softmax(z / t, -1)
        h = -(p * p.log()).sum()
        h.backward()
        assert math.isclose(t.grad.item(), dists.dH_dT(z, T).item(), rel_tol=1e-4)
        assert dists.dH_dT(z, T) >= 0


def test_temperature_limits_and_odds_ratio():
    z = torch.tensor([3.0, 2.0, 0.5, -1.0])
    assert dists.softmax(z, 0.01)[0] > 0.999
    assert torch.allclose(dists.softmax(z, 1e4), torch.full((4,), 0.25), atol=1e-3)
    r1, r_half = dists.odds_ratio(z, 1.0), dists.odds_ratio(z, 0.5)
    assert math.isclose(r_half, r1**2, rel_tol=1e-5)  # halving T squares the odds ratio
    assert math.isclose(r1, math.exp(-1.0), rel_tol=1e-5)


def test_effective_support_of_uniform():
    p = torch.full((32,), 1 / 32)
    assert math.isclose(dists.effective_support(p).item(), 32, rel_tol=1e-5)
    assert math.isclose(dists.entropy_bits(p).item(), 5, rel_tol=1e-5)


def _ref_sets(p, rule, param):
    """Plain-Python definitions of each candidate set, for comparison."""
    ps = p.tolist()
    order = sorted(range(len(ps)), key=lambda i: -ps[i])
    H = -sum(x * math.log(x) for x in ps if x > 0)
    if rule == "top_k":
        return set(order[:param])
    if rule == "top_p":
        out, mass = set(), 0.0
        for i in order:
            out.add(i)
            mass += ps[i]
            if mass >= param:
                return out
        return out
    if rule == "epsilon":
        return {i for i in range(len(ps)) if ps[i] > param} | {order[0]}
    if rule == "eta":
        eta = min(param, math.sqrt(param) * math.exp(-H))
        return {i for i in range(len(ps)) if ps[i] > eta} | {order[0]}
    if rule == "min_p":
        return {i for i in range(len(ps)) if ps[i] >= param * ps[order[0]]}
    if rule == "typical":
        typ = sorted(range(len(ps)), key=lambda i: abs(-math.log(ps[i]) - H))
        out, mass = set(), 0.0
        for i in typ:
            out.add(i)
            mass += ps[i]
            if mass >= param:
                return out
        return out


@pytest.mark.parametrize("rule,param", [("top_k", 5), ("top_p", 0.8), ("epsilon", 0.02), ("eta", 0.02),
                                        ("min_p", 0.2), ("typical", 0.7)])
def test_rules_match_reference(rule, param):
    torch.manual_seed(1)
    for _ in range(20):
        p = torch.softmax(torch.randn(40) * 2, -1)
        got = set(torch.nonzero(tr.RULES[rule](p[None], param)[0]).flatten().tolist())
        assert got == _ref_sets(p, rule, param)


def test_top_p_dominant_token_swallows_the_budget():
    p = torch.tensor([[0.93, 0.03, 0.02, 0.02]])
    assert tr.top_p_mask(p, 0.9).sum() == 1
    assert tr.epsilon_mask(p, 0.01).sum() == 4


def test_typical_can_reject_the_argmax():
    p = torch.tensor([[0.4] + [0.6 / 30] * 30])
    keep = tr.typical_mask(p, 0.5)
    assert not keep[0, 0]


def test_order_of_operations_changes_min_p_set():
    z = torch.linspace(4, -4, 100)[None]
    a = tr.candidate_mask(z, "min_p", 0.1, temperature=3.0, order="truncate_then_temper").sum()
    b = tr.candidate_mask(z, "min_p", 0.1, temperature=3.0, order="temper_then_truncate").sum()
    assert b > a


def test_sample_respects_candidate_set():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(1, 100).expand(500, -1)
    keep = tr.candidate_mask(z[:1], "top_k", 3)[0]
    draws = tr.sample(z, "top_k", 3, generator=g)
    assert keep[draws].all()
    assert (tr.sample(z, temperature=0) == z.argmax(-1)).all()


@pytest.mark.parametrize("rule,param,warper", [
    ("top_k", 7, "TopKLogitsWarper"), ("top_p", 0.8, "TopPLogitsWarper"), ("min_p", 0.1, "MinPLogitsWarper"),
    ("epsilon", 0.01, "EpsilonLogitsWarper"), ("eta", 0.01, "EtaLogitsWarper"), ("typical", 0.7, "TypicalLogitsWarper")])
def test_rules_match_hugging_face_warpers(rule, param, warper):
    from transformers.generation import logits_process as lp

    torch.manual_seed(2)
    z = torch.randn(16, 300) * 2.5
    ours = tr.RULES[rule](torch.softmax(z, -1), param)
    theirs = getattr(lp, warper)(param)(torch.zeros(16, 1, dtype=torch.long), z.clone()) > float("-inf")
    assert torch.equal(ours, theirs)
