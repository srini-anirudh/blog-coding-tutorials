"""Checks of the identities the post derives, on a policy small enough to enumerate."""

import pytest
import torch

from rlfp.advantages import gae, returns, terminal_rewards
from rlfp.estimators import baseline_weights, flat_grad, pg_loss
from rlfp.exact import ExactTask, all_sequences, exact_advantages, exact_J, pass_at_k, prefix_values, seq_logprobs
from rlfp.policy import TinyLM, prompt_batch, sample, token_logprobs
from rlfp.ppo import clip_region_table, clipped_surrogate, kl_estimators
from rlfp.tasks import exact_match_reward, sum_reward
from rlfp.tensors import chunked_token_logprobs, per_token_weight, resident_gb, response_mask


@pytest.fixture
def small():
    task = ExactTask(4, 4, sum_reward, [4], [6])
    torch.manual_seed(0)
    return task, TinyLM(5, 4, hidden=16)


def test_all_sequences():
    s = all_sequences(3, 4)
    assert s.shape == (81, 4) and len({tuple(r.tolist()) for r in s}) == 81


def test_distribution_normalised(small):
    task, pol = small
    assert seq_logprobs(pol, 4, task.seqs).exp().sum().item() == pytest.approx(1.0, abs=1e-5)


def test_sampled_logprob_matches_teacher_forced(small):
    task, pol = small
    acts = sample(pol, prompt_batch([4], 3), 4, torch.Generator().manual_seed(0))
    idx = (acts * torch.tensor([64, 16, 4, 1])).sum(1)
    lp_all = seq_logprobs(pol, 4, task.seqs)
    assert torch.allclose(token_logprobs(pol, prompt_batch([4], 3), acts).sum(1), lp_all[idx], atol=1e-5)


def test_score_function_identity_and_free_baseline(small):
    task, pol = small
    params = list(pol.parameters())
    logp = seq_logprobs(pol, 4, task.seqs)
    R = task.rewards[0]
    g_true = flat_grad(exact_J(logp, R), params, retain_graph=True)
    g_score = flat_grad((logp.detach().exp() * R * logp).sum(), params, retain_graph=True)
    g_base = flat_grad((logp.detach().exp() * logp).sum(), params)
    assert torch.allclose(g_true, g_score, atol=1e-7)
    assert g_base.norm() < 1e-5 * max(1.0, float(g_true.norm()))


def test_prefix_values_and_advantages(small):
    task, pol = small
    with torch.no_grad():
        logp = seq_logprobs(pol, 4, task.seqs)
    R = task.rewards[0]
    V = prefix_values(logp, R, task.seqs, 4)
    assert torch.allclose(V[:, -1], R.double())
    assert torch.allclose(V[:, 0], exact_J(logp, R).double().expand(len(R)), atol=1e-6)
    A = exact_advantages(V)
    p = logp.exp().double()
    # E_a[A(s, a)] = 0 at the initial state, and the advantages telescope to R - V(s_0).
    assert float((p * A[:, 0]).sum()) == pytest.approx(0.0, abs=1e-9)
    assert torch.allclose(A.sum(1), R.double() - V[:, 0])


def test_baseline_weights_by_hand():
    R = torch.tensor([1.0, 0.0, 0.0, 1.0])
    assert torch.allclose(baseline_weights(R, "loo"), torch.tensor([1 - 1 / 3, -2 / 3, -2 / 3, 1 - 1 / 3]))
    assert torch.allclose(baseline_weights(R, "group", 2), torch.tensor([1.0, -1.0, -1.0, 1.0]))
    assert torch.allclose(baseline_weights(R, "batch"), R - 0.5)
    assert torch.all(baseline_weights(torch.ones(4), "group", 2) == 0)  # equal rewards teach nothing


def test_pg_loss_is_reward_weighted_sft():
    lp = torch.randn(3, 5, requires_grad=True)
    w = torch.tensor([1.0, 0.0, -1.0])
    (g,) = torch.autograd.grad(pg_loss(lp, w), lp)
    assert torch.allclose(g, -w.unsqueeze(1).expand(3, 5) / 3)


def test_gae_limits():
    torch.manual_seed(0)
    R, V = torch.tensor([1.0, 0.0]), torch.rand(2, 5)
    r = terminal_rewards(R, 5)
    assert torch.allclose(gae(r, V, 1.0, 1.0), R.unsqueeze(1) - V, atol=1e-6)
    nextV = torch.cat([V[:, 1:], torch.zeros(2, 1)], 1)
    assert torch.allclose(gae(r, V, 1.0, 0.0), r + nextV - V, atol=1e-6)
    assert torch.all(returns(r) == R.unsqueeze(1))


def test_gae_mask_stops_at_eos():
    R, V = torch.tensor([1.0]), torch.tensor([[0.2, 0.4, 9.0, 9.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    adv = gae(terminal_rewards(R, 4, torch.tensor([2])), V, 1.0, 1.0, mask)
    assert torch.allclose(adv, torch.tensor([[0.8, 0.6, 0.0, 0.0]]))


def test_clip_regions():
    rows = {(r["A"], r["ratio"]): r["d_obj_d_ratio"] for r in clip_region_table(0.2)}
    assert rows[(1.0, 1.1)] == 1 and rows[(1.0, 1.4)] == 0
    assert rows[(-1.0, 0.9)] == -1 and rows[(-1.0, 0.6)] == 0 and rows[(-1.0, 1.4)] == -1


def test_clipped_surrogate_equals_pg_at_ratio_one():
    lp = torch.randn(4, 3, requires_grad=True)
    adv = torch.tensor([1.0, -1.0, 0.5, 2.0])
    loss, st = clipped_surrogate(lp, lp.detach(), adv, torch.ones(4, 3))
    (g1,) = torch.autograd.grad(loss, lp)
    (g2,) = torch.autograd.grad(pg_loss(lp, adv) / 3, lp)
    assert torch.allclose(g1, g2) and st["clip_frac"] == 0


def test_kl_estimators_unbiased_exactly():
    torch.manual_seed(0)
    lp, lq = torch.randn(10).log_softmax(0), torch.randn(10).log_softmax(0)
    true = float((lp.exp() * (lp - lq)).sum())
    est = kl_estimators(lp, lq)
    for k in ("k1", "k3"):
        assert float((lp.exp() * est[k]).sum()) == pytest.approx(true, rel=1e-5)
    assert torch.all(est["k3"] >= 0) and torch.any(est["k1"] < 0)


def test_cold_start_gradient_is_exactly_zero():
    torch.manual_seed(0)
    pol = TinyLM(17, 16, hidden=16)
    pb = prompt_batch([16], 32)
    acts = sample(pol, pb, 8, torch.Generator().manual_seed(0))
    R = exact_match_reward(acts, torch.tensor([3, 14, 1, 5, 9, 2, 6, 5]))
    assert R.sum() == 0
    pg_loss(token_logprobs(pol, pb, acts), baseline_weights(R, "group", 32)).backward()
    assert all(float(p.grad.abs().max()) == 0.0 for p in pol.parameters())


def test_pass_at_k():
    assert float(pass_at_k(0.5, 2)) == pytest.approx(0.75)
    assert float(pass_at_k(0.0, 64)) == 0.0


def test_response_mask_and_normalisation():
    acts = torch.tensor([[1, 10, 3, 10], [2, 2, 2, 2]])
    assert torch.equal(response_mask(acts, 10), torch.tensor([[1.0, 1, 0, 0], [1, 1, 1, 1]]))
    seq = per_token_weight([100, 2000], "seq")
    tok = per_token_weight([100, 2000], "token")
    assert seq[0] / seq[1] == pytest.approx(20) and tok[0] == pytest.approx(tok[1])


def test_chunked_logprobs_and_memory():
    torch.manual_seed(0)
    h, head, t = torch.randn(2, 37, 8), torch.nn.Linear(8, 50), torch.randint(0, 50, (2, 37))
    full = head(h).log_softmax(-1).gather(-1, t.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(full, chunked_token_logprobs(h, head, t, chunk=5), atol=1e-6)
    four = resident_gb(7e9, dict(policy=1, critic=1, reference=1, reward_model=1))
    assert four["total"] == pytest.approx(252) and four["policy"] == pytest.approx(112)
