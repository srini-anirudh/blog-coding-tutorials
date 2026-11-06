"""Bradley-Terry, the KL tilt and the DPO identity: each claim from the post as an assertion."""

import math

import torch

from prefopt.bt import (bt_grad_magnitude, bt_loss, bt_prob, best_logistic_match, choice_rate, fit_bt,
                        simulate_comparisons, thurstone_prob)
from prefopt.tilt import (fit_dpo_tabular, implicit_reward, kl, log_partition, objective, optimal_logp,
                          solve_by_gradient)


def test_gumbel_noise_gives_sigmoid_and_gaussian_gives_probit():
    gen = torch.Generator().manual_seed(0)
    for delta in (-1.0, 0.5, 2.0):
        assert abs(choice_rate(delta, "gumbel", 200_000, gen) - bt_prob(delta).item()) < 0.005
        assert abs(choice_rate(delta, "gaussian", 200_000, gen) - thurstone_prob(delta).item()) < 0.005


def test_logistic_and_probit_nearly_indistinguishable_after_rescaling():
    k, gap = best_logistic_match()
    assert 1.1 < k < 1.3 and gap < 0.01


def test_shift_invariance_and_noise_scale_sets_reward_scale():
    gen = torch.Generator().manual_seed(1)
    r = torch.linspace(-1.5, 1.5, 8, dtype=torch.float64)
    w, l = simulate_comparisons(r, 20_000, gen, scale=2.0)
    s = fit_bt(8, w, l, steps=1500)
    assert torch.allclose(bt_loss(s, w, l), bt_loss(s + 5.0, w, l))
    slope = (s * (r - r.mean())).sum() / ((r - r.mean()) ** 2).sum()
    assert abs(slope.item() - 0.5) < 0.08  # noise scale 2 halves the recovered reward scale


def test_bt_gradient_is_sigmoid_of_minus_delta():
    d = torch.linspace(-6, 6, 13, dtype=torch.float64)
    assert torch.allclose(bt_grad_magnitude(d), torch.sigmoid(-d))


def test_cycle_has_no_scalar_explanation():
    w = torch.tensor([0, 1, 2] * 9 + [1, 2, 0])
    l = torch.tensor([1, 2, 0] * 9 + [0, 1, 2])
    s = fit_bt(3, w, l, steps=1000)
    assert s.abs().max() < 1e-6 and abs(bt_loss(s, w, l).item() - math.log(2)) < 1e-6


def test_closed_form_beats_gradient_ascent_and_equals_beta_log_z():
    gen = torch.Generator().manual_seed(2)
    ref = torch.log_softmax(torch.randn(20, generator=gen, dtype=torch.float64), -1)
    r = torch.randn(20, generator=gen, dtype=torch.float64)
    beta = 0.5
    star = optimal_logp(ref, r, beta)
    j_star = objective(star, ref, r, beta)
    assert torch.allclose(j_star, beta * log_partition(ref, r, beta))
    found = solve_by_gradient(ref, r, beta, steps=1500)
    assert objective(found, ref, r, beta) <= j_star + 1e-12
    assert (j_star - objective(found, ref, r, beta)).item() < 1e-4


def test_support_is_preserved():
    ref = torch.log(torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64))
    r = torch.tensor([0.0, 1.0, 100.0], dtype=torch.float64)
    for beta in (10.0, 0.1, 0.001):
        p = optimal_logp(ref, r, beta).exp()
        assert p[2] == 0.0
        assert torch.isfinite(kl(optimal_logp(ref, r, beta), ref))


def test_implicit_reward_recovers_reward_up_to_per_prompt_constant():
    gen = torch.Generator().manual_seed(3)
    ref = torch.log_softmax(torch.randn(4, 5, generator=gen, dtype=torch.float64), -1)
    r = torch.randn(4, 5, generator=gen, dtype=torch.float64)
    beta = 0.3
    gap = implicit_reward(optimal_logp(ref, r, beta), ref, beta) - r
    assert torch.allclose(gap, -beta * log_partition(ref, r, beta).unsqueeze(-1).expand_as(gap))


def test_dpo_on_exact_preferences_recovers_optimal_policy():
    """Every pair, many times, with BT labels: tabular DPO should land on pi* = ref exp(r / beta) / Z."""
    gen = torch.Generator().manual_seed(4)
    P, K, beta = 2, 4, 0.5
    ref = torch.log_softmax(torch.randn(P, K, generator=gen, dtype=torch.float64), -1)
    r = torch.randn(P, K, generator=gen, dtype=torch.float64)
    x, w, l = [], [], []
    for p in range(P):
        for i in range(K):
            for j in range(K):
                if i != j:  # weight each ordered pair by its BT probability, via repetition
                    reps = round(1000 * bt_prob(r[p, i] - r[p, j]).item())
                    x += [p] * reps
                    w += [i] * reps
                    l += [j] * reps
    pi = fit_dpo_tabular(ref, torch.tensor(x), torch.tensor(w), torch.tensor(l), beta, steps=3000)
    tv = 0.5 * (pi.exp() - optimal_logp(ref, r, beta).exp()).abs().sum(-1).max()
    assert tv < 0.01
