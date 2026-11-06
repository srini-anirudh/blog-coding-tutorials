import numpy as np
import torch

from rlvrlab.memory import GRPO_RLVR, PPO_RLHF, measure_bytes_per_param, persistent_state_gb
from rlvrlab.prm import hard_label_good_prob, label_accuracy, mc_step_labels, sample_derivations
from rlvrlab.verifier import best_of_n, extract_answer, format_reward, math_reward, surface_proxy


def test_extract_answer_prefers_last_boxed():
    assert extract_answer("x \\boxed{3} then \\boxed{4}") == "4"
    assert extract_answer("so the answer is 1,024.") == "1,024"
    assert extract_answer("no answer here") is None


def test_math_reward_normalizes_numbers():
    assert math_reward("\\boxed{42.0}", "42") == 1.0
    assert math_reward("\\boxed{1,000}", "1000") == 1.0
    assert math_reward("\\boxed{41}", "42") == 0.0


def test_format_reward_needs_nonempty_think_and_an_answer():
    assert format_reward("<think>a</think> b") == 1.0
    assert format_reward("<think> </think> b") == 0.0
    assert format_reward("<think>a</think>") == 0.0
    assert format_reward("plain text") == 0.0


def test_surface_proxy_ignores_correctness():
    a = "First clearly 6 x 7.\nThen therefore.\nFinally \\boxed{48}"
    assert surface_proxy(a) == surface_proxy(a.replace("48", "42"))


def test_best_of_n_oracle_monotone_and_scorer_turns_over():
    ns = [1, 4, 32, 1024]
    c = best_of_n(ns, n_problems=1000)
    assert all(x <= y for x, y in zip(c["oracle"], c["oracle"][1:]))
    assert c["scorer"][2] > c["scorer"][0] and c["scorer"][3] < c["scorer"][2]
    clean = best_of_n(ns, n_problems=1000, fool_rate=0.0)
    assert clean["scorer"][3] > clean["scorer"][2]


def test_mc_hard_labels_match_closed_form():
    step_ok = np.zeros((200_000, 1), dtype=bool)
    hard, _ = mc_step_labels(step_ok, a_clean=0.9, a_rescue=0.3, m=4)
    assert abs(hard.mean() - hard_label_good_prob(0.3, 4)) < 0.005


def test_strong_completer_hides_bad_steps():
    step_ok, _ = sample_derivations(20_000, 6, 0.1, 0.1)
    weak = label_accuracy(step_ok, mc_step_labels(step_ok, 0.5, 0.02, 8)[0])
    strong = label_accuracy(step_ok, mc_step_labels(step_ok, 0.95, 0.4, 8)[0])
    assert strong["bad_steps_called_bad"] < 0.1 < weak["bad_steps_called_bad"]


def test_ppo_memory_matches_post():
    ppo = persistent_state_gb(PPO_RLHF, 7e9)
    assert round(ppo["total"]) == 252 and abs(ppo["critic"] / ppo["total"] - 0.444) < 0.001
    assert round(persistent_state_gb(GRPO_RLVR, 7e9)["total"]) == 126


def test_sixteen_bytes_per_trained_param():
    for mixed in (False, True):
        m = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16))
        assert measure_bytes_per_param(m, mixed_precision=mixed) == 16.0
