"""The loss family, likelihood displacement in the toy policy, and the Goodhart toy."""

import math

import torch

from prefopt.bandit import bt_margin_grad, policy_logp, similar_pairs_world, train_dpo
from prefopt.losses import FAMILY, dpo_sequence_loss, loss_and_grad
from prefopt.overopt import best_of_n_curve, bon_kl, fit_form, fit_proxy, gold, tilt_curve

H = torch.linspace(-4, 8, 25, dtype=torch.float64)


def test_dpo_gradient_closed_form():
    for beta in (0.1, 1.0, 3.0):
        _, g = loss_and_grad("DPO", H, beta)
        assert torch.allclose(g, -beta * torch.sigmoid(-beta * H))


def test_hinges_stop_and_ipo_pulls_back():
    _, g = loss_and_grad("SLiC hinge", H, 1.0, 1.6)
    assert (g[H > 1.6] == 0).all() and (g[H < 1.6] == -1).all()
    _, g = loss_and_grad("IPO", H, 1.0)
    assert (g[H > 0.5] > 0).all() and (g[H < 0.5] < 0).all()


def test_simpo_is_dpo_shifted_by_target_margin():
    beta, target = 2.0, 1.0
    l_simpo, _ = loss_and_grad("SimPO", H, beta, target)
    l_dpo, _ = loss_and_grad("DPO", H - target, beta)
    assert torch.allclose(l_simpo, l_dpo)


def test_every_family_member_pushes_a_wrong_pair_the_right_way():
    for name in FAMILY:
        _, g = loss_and_grad(name, torch.tensor([-1.0], dtype=torch.float64))
        assert g.item() < 0, name


def test_dpo_gradient_formula_in_toy_policy():
    """grad = -beta sigma(r_l - r_w) (grad log pi_w - grad log pi_l); with a softmax over phi.theta, grad log pi_y =
    phi_y - E_pi[phi], so the expectation cancels and grad = -beta sigma(...) (phi_w - phi_l)."""
    world, w, l = similar_pairs_world(1, 20, 6, 0.3)
    theta = torch.randn(6, dtype=torch.float64) * 0.3
    beta = 0.7
    g = bt_margin_grad(world, theta, w, l, beta)
    logp = policy_logp(world, theta)
    r_w, r_l = beta * (logp[w] - world.ref_logp[w]), beta * (logp[l] - world.ref_logp[l])
    expect = -beta * torch.sigmoid(r_l - r_w) * (world.phi[w] - world.phi[l])
    assert torch.allclose(g, expect.squeeze(0))


def test_likelihood_displacement_loss_falls_while_chosen_falls():
    world, w, l = similar_pairs_world(8, 300, 32, 0.9)
    _, h = train_dpo(world, w, l, beta=0.1, steps=150, lr=0.05, log_every=50)
    assert h[-1]["loss"] < h[0]["loss"] - 0.1
    assert h[-1]["d_logp_chosen"] < -1.0
    assert h[-1]["mass_elsewhere"] > h[0]["mass_elsewhere"]


def test_dpop_keeps_chosen_at_reference():
    world, w, l = similar_pairs_world(8, 300, 32, 0.9)
    _, h = train_dpo(world, w, l, beta=0.1, steps=150, lr=0.05, log_every=150, dpop_lambda=5.0)
    assert h[-1]["d_logp_chosen"] > -0.5


def test_dpop_and_nll_reduce_to_dpo_when_off():
    a = torch.randn(4, dtype=torch.float64)
    b, c, d = a - 1, a + 0.2, a - 0.3
    base = -torch.nn.functional.logsigmoid(0.5 * ((a - c) - (b - d))).mean()
    assert torch.allclose(dpo_sequence_loss(a, b, c, d, 0.5), base)
    assert dpo_sequence_loss(a, b, c, d, 0.5, dpop_lambda=3.0) > base  # chosen below reference: penalised


def test_bon_kl_formula():
    assert bon_kl(1) == 0.0
    assert math.isclose(bon_kl(4), math.log(4) - 0.75)


def test_fit_form_recovers_known_coefficients():
    d = torch.linspace(0.2, 3, 30, dtype=torch.float64)
    a, b, r2 = fit_form(d, d * (2.0 - 0.5 * d), "bo")
    assert abs(a - 2.0) < 1e-9 and abs(b - 0.5) < 1e-9 and r2 > 0.999999
    a, b, r2 = fit_form(d, d * (1.5 - 0.3 * torch.log(d)), "rl")
    assert abs(a - 1.5) < 1e-9 and abs(b - 0.3) < 1e-9


def test_goodhart_linear_proxy_rises_then_falls():
    proxy = fit_proxy(500, 0, epochs=200)
    rows = best_of_n_curve(proxy, [1, 8, 64, 4096], trials=200)
    golds = [r["gold"] for r in rows]
    proxies = [r["proxy"] for r in rows]
    assert proxies == sorted(proxies)  # the proxy keeps improving
    assert golds[2] > golds[0] and golds[3] < golds[2]  # gold peaks, then falls


def test_tilt_curve_ess_and_kl_grow_together():
    proxy = fit_proxy(500, 0, epochs=200)
    rows = tilt_curve(proxy, [4.0, 1.0, 0.5], pool=50_000)
    ds = [r["d"] for r in rows]
    assert ds == sorted(ds) and rows[0]["ess"] > rows[-1]["ess"]
    assert abs(gold(torch.zeros(1, 4, dtype=torch.float64)).item()) < 1e-12
