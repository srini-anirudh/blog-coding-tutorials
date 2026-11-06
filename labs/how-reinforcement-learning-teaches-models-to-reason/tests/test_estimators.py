import math

import pytest
import torch

from rlvrlab.advantages import (advantage_mass, binary_closed_form, dynamic_sampling_mask, group_advantages,
                                informative_prob)
from rlvrlab.bandit import estimator_stats, exact_gradient, group_gradients

THETA = torch.tensor([0.5, 1.0, 0.0, -0.5], dtype=torch.float64)
C = torch.tensor([0.9, 0.5, 0.2, 0.0], dtype=torch.float64)


def test_exact_gradient_matches_autograd():
    t = THETA.clone().requires_grad_(True)
    (t.softmax(-1) @ C).backward()
    assert torch.allclose(t.grad, exact_gradient(THETA, C))


def test_group_gradients_match_one_hot_formula():
    a = torch.tensor([[0, 1, 1, 3]])
    adv = torch.tensor([[1.0, -0.5, 2.0, 0.3]], dtype=torch.float64)
    onehot = torch.nn.functional.one_hot(a, 4).double()
    want = (adv.unsqueeze(-1) * (onehot - THETA.softmax(-1))).mean(1)
    assert torch.allclose(group_gradients(THETA, a, adv), want)


def test_rloo_is_leave_one_out():
    r = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    want = torch.tensor([[1 - 2 / 3, 0 - 1.0, 1 - 2 / 3, 1 - 2 / 3]])
    assert torch.allclose(group_advantages(r, "rloo"), want)


@pytest.mark.parametrize("g", [2, 4, 8])
def test_baselines_scale(g):
    mean = estimator_stats(THETA, C, "mean", g, groups=100_000)
    rloo = estimator_stats(THETA, C, "rloo", g, groups=100_000)
    const = estimator_stats(THETA, C, "reinforce", g, groups=100_000, baseline=0.5)
    tol = 5 * mean["mc_stderr"] + 0.01
    assert abs(mean["scale"] - (g - 1) / g) < tol  # group mean shrinks the gradient by (G-1)/G
    assert abs(rloo["scale"] - 1) < 5 * rloo["mc_stderr"] + 0.01  # leave-one-out is unbiased
    assert abs(const["scale"] - 1) < 5 * const["mc_stderr"] + 0.01  # any constant baseline is free


def test_baseline_reduces_variance():
    plain = estimator_stats(THETA, C, "reinforce", 8, groups=50_000)
    rloo = estimator_stats(THETA, C, "rloo", 8, groups=50_000)
    assert rloo["rel_variance"] < plain["rel_variance"]


def test_binary_closed_form_matches_tensor_and_post_numbers():
    for k in range(1, 8):
        r = torch.zeros(1, 8)
        r[0, :k] = 1
        a = group_advantages(r, "grpo", eps=0.0)
        right, wrong = binary_closed_form(k / 8)
        assert math.isclose(float(a[0, 0]), right, rel_tol=1e-5) and math.isclose(float(a[0, -1]), wrong, rel_tol=1e-5)
    right, wrong = binary_closed_form(7 / 8)
    assert round(right, 2) == 0.38 and round(wrong, 2) == -2.65


def test_unanimous_groups_have_no_gradient():
    r = torch.tensor([[1.0] * 8, [0.0] * 8, [1.0] + [0.0] * 7])
    assert group_advantages(r, "grpo")[:2].abs().max() < 1e-6
    assert dynamic_sampling_mask(r).tolist() == [False, False, True]
    assert informative_prob(0.5, 8) == 1 - 2 * 0.5**8


def test_advantage_mass_closed_forms():
    for k in range(1, 8):
        p = k / 8
        assert math.isclose(advantage_mass(k, 8, "grpo"), 16 * math.sqrt(p * (1 - p)), rel_tol=1e-5)
        assert math.isclose(advantage_mass(k, 8, "mean"), 16 * p * (1 - p), rel_tol=1e-5)
