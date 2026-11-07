import math

import pytest
import torch

from prefopt.bt import best_logistic_rescale, bt_loss, fit_scores, probit, sample_comparisons, simulate_choices
from prefopt.tilt import best_of_n, kl, log_partition, maximise_numerically, objective, tilt


def test_gumbel_noise_gives_sigmoid_and_gaussian_gives_probit():
    d = torch.tensor([-2.0, 0.0, 1.0, 2.5], dtype=torch.float64)
    assert (simulate_choices(d, 100_000, "gumbel") - torch.sigmoid(d)).abs().max() < 0.006
    assert (simulate_choices(d, 100_000, "gaussian") - probit(d)).abs().max() < 0.006


def test_noise_scale_divides_the_gap():
    d = torch.tensor([1.5], dtype=torch.float64)
    rate = simulate_choices(d, 100_000, "gumbel", scale=2.0)
    assert abs(rate.item() - torch.sigmoid(d / 2).item()) < 0.006


def test_logistic_and_probit_nearly_agree_after_rescaling():
    c, gap = best_logistic_rescale(torch.linspace(-6, 6, 601))
    assert abs(c - 1.702 / math.sqrt(2)) < 0.02 and gap < 0.01


def test_bt_loss_is_shift_invariant_and_its_gradient_anneals():
    a, b = torch.randn(50), torch.randn(50)
    assert torch.allclose(bt_loss(a, b), bt_loss(a + 3.7, b + 3.7))
    d = torch.linspace(-4, 4, 9, requires_grad=True)
    (g,) = torch.autograd.grad(-torch.nn.functional.logsigmoid(d).sum(), d)
    assert torch.allclose(g, -torch.sigmoid(-d.detach()), atol=1e-6)


def test_fit_recovers_scores_up_to_a_constant():
    r = torch.linspace(-1.5, 1.5, 8)
    w, l = sample_comparisons(r, 20_000, seed=0)
    f = fit_scores(w, l, 8, steps=800)
    assert torch.corrcoef(torch.stack([f, r]))[0, 1] > 0.995


def test_tilt_is_the_maximiser_and_its_value_is_beta_log_z():
    g = torch.Generator().manual_seed(0)
    q = torch.softmax(torch.randn(6, generator=g, dtype=torch.float64), -1)
    r = torch.randn(6, generator=g, dtype=torch.float64)
    beta = 0.5
    p = tilt(q, r, beta)
    assert torch.isclose(objective(p, q, r, beta), beta * log_partition(q, r, beta))
    num = maximise_numerically(q, r, beta, steps=1500)
    assert (num - p).abs().max() < 1e-4
    for _ in range(200):
        other = torch.softmax(torch.randn(6, generator=g, dtype=torch.float64), -1)
        assert objective(other, q, r, beta) <= objective(p, q, r, beta) + 1e-12


def test_tilt_keeps_zero_support_at_zero():
    q = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    r = torch.tensor([0.0, 1.0, 100.0], dtype=torch.float64)
    assert tilt(q, r, 0.01)[2] == 0
    assert kl(torch.tensor([0.0, 0.0, 1.0]), q) == float("inf")


@pytest.mark.parametrize("n", [1, 2, 7])
def test_best_of_n_matches_simulation(n):
    q = torch.tensor([0.1, 0.4, 0.3, 0.2], dtype=torch.float64)
    score = torch.tensor([3.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    p = best_of_n(q, score, n)
    g = torch.Generator().manual_seed(1)
    draws = torch.multinomial(q, 200_000 * n, replacement=True, generator=g).view(-1, n)
    best = draws.gather(1, score[draws].argmax(1, keepdim=True)).squeeze(1)
    emp = torch.bincount(best, minlength=4).double() / best.numel()
    assert (emp - p).abs().max() < 0.005
    assert torch.isclose(p.sum(), torch.tensor(1.0, dtype=torch.float64))
