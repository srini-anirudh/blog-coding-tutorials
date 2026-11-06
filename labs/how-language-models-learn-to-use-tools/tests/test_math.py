import math
import random

import pytest
import torch

from toollab.compound import required_p, success_exact, success_monte_carlo
from toollab.evaluate import score, summarise
from toollab.grammar import call_mask, local_vs_global
from toollab.model import build_model, decode, train
from toollab.toolformer import filter_call, toolformer_weights, weighted_loss
from toollab.verify import format_ok
from toollab.world import Vocab, make_example, make_tools, parse_call, split_names


def test_product_law_and_requirement():
    assert success_exact(0.95, 0.0, 10) == pytest.approx(0.95 ** 10)
    assert required_p(0.9, 10) == pytest.approx(0.98952, abs=1e-5)
    assert required_p(0.9, 50) == pytest.approx(0.99790, abs=1e-5)
    assert success_exact(0.95, 0.5, 50) > success_exact(0.96, 0.0, 50)


def test_monte_carlo_matches_matrix_power():
    assert success_monte_carlo(0.9, 0.5, 10, 20_000) == pytest.approx(success_exact(0.9, 0.5, 10), abs=0.015)


def test_toolformer_weights():
    w = toolformer_weights()
    assert torch.allclose(w, torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2], dtype=torch.float64) / 3.0)


class CharTok:
    bos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + (ord(c) % 60) for c in text]}


def test_weighted_loss_matches_a_hand_computation():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=128, n_embd=16, n_layer=1, n_head=2)).eval()
    tok = CharTok()
    got = weighted_loss(model, tok, "abc", "defghij")
    ids = [0] + tok("abc")["input_ids"] + tok("defgh")["input_ids"]
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.tensor([ids])).logits[0].double(), -1)
    nll = [-logp[3 + j, ids[4 + j]] for j in range(5)]
    assert got == pytest.approx(float(sum(w * n for w, n in zip(toolformer_weights(), nll))), rel=1e-6)
    r = filter_call(model, tok, "abc", "defg", "C(1)", "2")
    assert r["L_minus"] == min(r["L_empty"], r["L_call_no_result"])
    assert r["gain"] <= r["gain_without_min"]


def test_alpha_correction_recovers_exact_conditional():
    table = {(): {"A": 0.6, "B": 0.4}, ("A",): {"x": 0.9, "y": 0.1}, ("B",): {"x": 0.95, "z": 0.05}}
    valid = {("A", "y", "$"), ("B", "x", "$")}
    res = local_vs_global(lambda p: {"$": 1.0} if len(p) == 2 else table[p], lambda s: s in valid)
    for s in valid:
        assert res["alpha_corrected"][s] == pytest.approx(res["exact"][s])
    assert res["local"][("A", "y", "$")] == pytest.approx(0.6)
    assert res["exact"][("A", "y", "$")] == pytest.approx(0.06 / 0.44)
    assert res["kl_exact_to_local"] > 0


def tools(n=40):
    train_names, _ = split_names(seed=0)
    return make_tools(n, train_names, random.Random(0))


def test_mask_makes_an_untrained_model_always_parseable():
    torch.manual_seed(0)
    vocab, pool, rng = Vocab(), tools(), random.Random(1)
    model = build_model(len(vocab), hidden=32, layers=1, heads=2)
    for _ in range(20):
        ex = make_example(pool, rng)
        masked = decode(model, vocab, ex, max_new=40, mask=call_mask(ex.tools, typed=True))
        assert format_ok(ex, parse_call(masked))


def test_training_reduces_loss_and_scoring_runs():
    vocab, pool = Vocab(), tools(8)
    model = build_model(len(vocab), hidden=32, layers=1, heads=2)
    hist = train(model, pool, vocab, steps=60, batch=16, lr=3e-3, log_every=20, log=lambda s: None)
    assert hist[-1][1] < hist[0][1]
    ex = make_example(pool, random.Random(9))
    s = summarise([score(ex, decode(model, vocab, ex))])
    assert s["n"] == 1 and not math.isnan(s["parse"])
