import torch

from rlvrlab.length_bias import (LengthConfig, expected_stop_gradient, response_logprob, sample_lengths,
                                 stop_gradient_bias, train)
from rlvrlab.operators import run_operator, sft_grad_as_policy_gradient, sft_grad_exact, token_weights
from rlvrlab.sharpening import (SharpenConfig, base_logits, make_problems, pass_at_k_exact, pass_at_k_unbiased)
from rlvrlab.sharpening import train as train_sharpen


def test_response_logprob_matches_geometric_pmf():
    z = torch.tensor([-2.0, 0.0])
    q = torch.sigmoid(z[0])
    L = torch.tensor([1.0, 3.0, 5.0])
    lp = response_logprob(z, torch.zeros(3, dtype=torch.long), L, torch.zeros(3, dtype=torch.bool))
    assert torch.allclose(lp, torch.log(q * (1 - q) ** (L - 1)))


def test_sampled_lengths_have_mean_one_over_q():
    gen = torch.Generator().manual_seed(0)
    z = torch.logit(torch.tensor([0.05, 0.05]))
    L, _ = sample_lengths(z, torch.zeros(200_000, dtype=torch.long), 10_000, gen)
    assert abs(float(L.mean()) - 20) < 0.3


def test_jensen_sign_of_length_push():
    biased = expected_stop_gradient(-1, 1, 1 / 40, "grpo", 40)
    unbiased = expected_stop_gradient(-1, 1, 1 / 40, "drgrpo", 40)
    assert biased["wrong"] < 0 < biased["right"]
    assert abs(unbiased["wrong"]) < 1e-12


def test_grpo_bias_is_real_and_drgrpo_is_not():
    g = stop_gradient_bias(LengthConfig(loss="grpo"), groups=20_000)
    d = stop_gradient_bias(LengthConfig(loss="drgrpo"), groups=20_000)
    assert g["guess"] / g["guess_stderr"] < -20
    assert abs(d["guess"] / d["guess_stderr"]) < 4


def test_grpo_training_lengthens_wrong_answers():
    biased = train(LengthConfig(loss="grpo", steps=100))
    fixed = train(LengthConfig(loss="drgrpo", steps=100))
    assert biased[-1]["len_wrong"] > 2 * biased[0]["len_wrong"]
    assert fixed[-1]["len_wrong"] < 1.5 * fixed[0]["len_wrong"]


def test_pass_at_k_estimator():
    assert pass_at_k_unbiased(10, 0, 3) == 0.0
    assert pass_at_k_unbiased(10, 8, 3) == 1.0
    assert abs(pass_at_k_unbiased(10, 1, 1) - 0.1) < 1e-12


def test_rl_sharpens_and_loses_coverage():
    cfg = SharpenConfig(steps=100, problems=500)
    tr, te = make_problems(cfg, 1), make_problems(cfg, 2)
    theta, _ = train_sharpen(cfg, tr, te, [1])
    base = pass_at_k_exact(base_logits(cfg), te, [1, 256])
    rl = pass_at_k_exact(theta, te, [1, 256])
    assert rl[1] > base[1] and rl[256] < base[256]


def test_sft_is_policy_gradient_with_inverse_prob_reward():
    theta = torch.tensor([1.0, 0.0, -1.0, 0.5], dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    est = sft_grad_as_policy_gradient(theta, 2, 400_000, gen)
    exact = sft_grad_exact(theta, 2)
    assert (est - exact).norm() / exact.norm() < 0.02
    w = token_weights(torch.tensor([0.001]))
    assert torch.isclose(w["implicit_reward_sft"], torch.tensor(1000.0))


def test_rl_cannot_reach_unsampled_mode_and_sft_can():
    theta = torch.tensor([1.0, 0.5, 0.0, -14.0], dtype=torch.float64)
    rewarded = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    sft, _ = run_operator(theta, "sft", 3, rewarded, 200, 1.0)
    rl, seen = run_operator(theta, "rl", 3, rewarded, 200, 1.0)
    assert sft.softmax(-1)[3] > 0.9
    assert seen == 0 and rl.softmax(-1)[3] <= theta.softmax(-1)[3]
