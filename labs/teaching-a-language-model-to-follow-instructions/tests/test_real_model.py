"""DPO plumbing on the real SmolLM2-135M-Instruct, reusing tutorial C6's pairs and log-prob code. Downloads."""

import pytest
import torch

pytestmark = pytest.mark.network


def test_pairs_are_near_identical_and_one_dpo_step_moves_the_margin():
    from prefopt.lm import build_examples, load, train

    torch.manual_seed(0)
    tok, policy = load("cpu")
    examples = build_examples(tok, 16)
    ex = examples[0]
    same = sum(a == b for a, b in zip(ex["chosen_ids"], ex["rejected_ids"]))
    assert same >= len(ex["chosen_ids"]) - 4  # one number differs
    hist = train(policy, tok, examples, "cpu", steps=3, batch=4, lr=1e-5, eval_every=3, n_eval=8, log=lambda r: None)
    assert hist[0]["d_logp_chosen"] == 0.0 and abs(hist[0]["dpo_loss"] - 0.6931) < 1e-3
    assert hist[-1]["dpo_loss"] < hist[0]["dpo_loss"]
