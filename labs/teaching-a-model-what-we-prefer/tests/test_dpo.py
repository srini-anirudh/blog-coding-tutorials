import pytest
import torch

from prefopt.dpo import FAMILY, SoftmaxPolicy, dpo_loss, dpo_weight, loss_ipo, loss_slic, train_dpo
from prefopt.tilt import log_partition, tilt


def test_implicit_reward_recovers_r_up_to_beta_log_z():
    g = torch.Generator().manual_seed(0)
    q = torch.softmax(torch.randn(8, generator=g, dtype=torch.float64), -1)
    r = torch.randn(8, generator=g, dtype=torch.float64)
    beta = 0.3
    rec = beta * (tilt(q, r, beta).log() - q.log())
    assert torch.allclose(r - rec, beta * log_partition(q, r, beta).expand(8))


def test_dpo_on_all_pairs_converges_to_the_tilt():
    g = torch.Generator().manual_seed(1)
    k, beta = 6, 0.5
    q = torch.softmax(torch.randn(k, generator=g, dtype=torch.float64), -1)
    r = torch.randn(k, generator=g, dtype=torch.float64)
    i, j = torch.triu_indices(k, k, 1)
    p = torch.sigmoid(r[i] - r[j])
    pol = SoftmaxPolicy(q.log())
    train_dpo(pol, torch.cat([i, j]), torch.cat([j, i]), beta, 1500, 0.05, weight=torch.cat([p, 1 - p]))
    assert (pol.log_probs().exp() - tilt(q, r, beta)).abs().max() < 1e-6


def test_matches_c06_dpo_loss():
    from prefopt.lm import c06_dpo_loss

    a, b, c, d = (torch.randn(5) for _ in range(4))
    assert torch.allclose(dpo_loss(a, b, c, d, 0.2), c06_dpo_loss(a, b, c, d, 0.2)[0])


def test_gradient_formula_on_tiny_llama():
    from importlib import util
    from pathlib import Path

    spec = util.spec_from_file_location("s03", Path(__file__).resolve().parents[1] / "scripts" / "03_dpo_identity.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rel, w, _ = mod.gradient_check()
    assert rel < 1e-5 and 0 < w < 1


def test_weight_is_large_when_the_pair_is_backwards():
    z = torch.zeros(1)
    assert dpo_weight(torch.tensor([-5.0]), z, z, z, 1.0) > 0.99
    assert dpo_weight(torch.tensor([5.0]), z, z, z, 1.0) < 0.01


def _pair(cos, seed=0, v=30, d=8):
    g = torch.Generator().manual_seed(seed)
    e = torch.nn.functional.normalize(torch.randn(v, d, generator=g, dtype=torch.float64), dim=-1)
    u = e[1] - (e[1] @ e[0]) * e[0]
    e[1] = cos * e[0] + (1 - cos**2) ** 0.5 * u / u.norm()
    return e, e @ torch.randn(d, generator=g, dtype=torch.float64)


def test_likelihood_displacement_for_similar_pairs_and_the_rpo_fix():
    w, l = torch.tensor([0]), torch.tensor([1])
    down = 0
    for seed in range(10):
        e, base = _pair(0.98, seed)
        pol = SoftmaxPolicy(base, e)
        ref = pol.log_probs().detach()
        logs = train_dpo(pol, w, l, 0.1, 200, 0.05, log_every=200)
        assert logs[-1]["loss"] < logs[0]["loss"]  # the DPO loss always improves ...
        down += int(pol.log_probs()[0] < ref[0])  # ... while the winner usually loses probability
        fixed = SoftmaxPolicy(base, e)
        train_dpo(fixed, w, l, 0.1, 200, 0.05, variant="rpo", lam=1.0)
        assert fixed.log_probs()[0] > ref[0]
    assert down >= 8


@pytest.mark.parametrize("name", list(FAMILY))
def test_family_losses_decrease_with_margin_near_zero(name):
    m = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    loss = FAMILY[name](m)
    assert loss[0] > loss[1] - 1e-12


def test_ipo_pulls_back_and_slic_stops():
    m = torch.tensor([3.0], requires_grad=True)
    (gi,) = torch.autograd.grad(loss_ipo(m).sum(), m)
    (gs,) = torch.autograd.grad(loss_slic(m).sum(), m)
    assert gi.item() > 0 and gs.item() == 0
