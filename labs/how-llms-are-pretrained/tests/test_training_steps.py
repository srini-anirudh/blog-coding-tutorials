"""Packing masks, the tiny LM, mixture fitting, repetition and signal/noise: offline and fast."""

import numpy as np
import pytest
import torch
from ptlab import evalnoise, mixing, repetition, tinylm
from ptlab.packing import compare


@pytest.fixture(scope="module")
def docs():
    rng = np.random.default_rng(0)
    return [rng.integers(1, 256, size=n).tolist() + [0] for n in (9, 12, 15)]


def test_packing_rope_needs_only_the_document_mask(docs):
    model = tinylm.make_model("xs", seq_len=64).eval()
    res = compare(model, docs, eos_id=0)
    assert max(res["document mask, positions reset"]) < 1e-5
    assert max(res["document mask, positions keep counting"]) < 1e-4  # RoPE: relative offsets only
    assert max(res["causal mask, positions keep counting"]) > 1e-3


def test_packing_absolute_positions_need_the_reset_too(docs):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=256, n_positions=64, n_embd=64, n_layer=2, n_head=2, bos_token_id=0, eos_token_id=0)
    cfg._attn_implementation = "eager"
    model = GPT2LMHeadModel(cfg).eval()
    res = compare(model, docs, eos_id=0)
    assert max(res["document mask, positions reset"]) < 1e-5
    assert max(res["document mask, positions keep counting"]) > 1e-3


def test_train_run_records_checkpoints():
    rng = np.random.default_rng(0)
    train = {"a": rng.integers(1, 256, 5000).astype(np.uint8), "b": rng.integers(1, 256, 5000).astype(np.uint8)}
    val = {d: tinylm.eval_windows(v, 4, 32) for d, v in train.items()}
    cfg = tinylm.RunConfig(weights={"a": 0.7, "b": 0.3}, size="xs", steps=6, batch=4, seq_len=32, warmup=2,
                           eval_every=3, last_k=3, last_every=1)
    r = tinylm.train_run(cfg, train, val, "cpu")
    assert [c["step"] for c in r["checkpoints"]] == [3, 4, 5, 6]
    assert all(len(c["bpb"]["a"]) == 4 for c in r["checkpoints"])
    assert r["checkpoints"][0]["bpb"]["a"][0] == pytest.approx(8.0, abs=0.5)  # near-uniform over 256 bytes


def test_choice_logprob_and_items():
    prompts, options, answers = tinylm.math_choice_items(5)
    for p, o, a in zip(prompts, options, answers):
        x, y = [int(t) for t in p.split(" + ")[0:1] + p.split(" + ")[1].split(" = ")[:1]]
        assert int(o[a]) == x + y
    s = tinylm.choice_logprob(tinylm.make_model("xs", 64).eval(), prompts, options, "cpu")
    assert s.shape == (5, 4) and np.all(s < 0)


def test_quadratic_surrogate_recovers_a_quadratic():
    w = mixing.sample_simplex(40, 3, seed=1)
    y = 1 + (w[:, 0] - 0.5) ** 2 + 0.3 * w[:, 1] * w[:, 2]
    m = mixing.QuadraticSurrogate(ridge=1e-9).fit(w, y)
    g = mixing.simplex_grid(3, 0.05)
    assert np.allclose(g.sum(1), 1)
    assert np.abs(m.predict(g) - (1 + (g[:, 0] - 0.5) ** 2 + 0.3 * g[:, 1] * g[:, 2])).max() < 1e-5


def test_mixing_law_fits_its_own_form():
    w = mixing.sample_simplex(25, 3, seed=2)
    t = np.array([[-2.0, 0.5, 0.3], [0.2, -1.5, 0.1]])
    L = np.column_stack([1.0 + 0.8 * np.exp(w @ t[0]), 2.0 + 0.5 * np.exp(w @ t[1])])
    law = mixing.MixingLaw().fit(w, L)
    assert np.abs(law.predict(w) - L).max() < 1e-3


def test_effective_tokens():
    assert repetition.effective_tokens(100, 1) == pytest.approx(100)
    assert repetition.value_fraction(4) > 0.9
    assert repetition.value_fraction(40) < 0.4
    assert repetition.effective_tokens(1, 1e6) == pytest.approx(1 + repetition.R_STAR_PAPER)
    u, e = np.array([1e6] * 4), np.array([2.0, 4, 16, 64])
    assert repetition.fit_r_star(u, e, repetition.effective_tokens(u, e, 15.4)) == pytest.approx(15.4, rel=0.01)


def test_interp_fresh():
    b, loss = np.array([1e3, 1e4, 1e5]), np.array([3.0, 2.0, 1.0])
    assert repetition.interp_fresh(b, loss, 1.5) == pytest.approx(10 ** 4.5, rel=1e-6)
    assert np.isnan(repetition.interp_fresh(b, loss, 0.5))


def test_signal_noise_and_decisions():
    assert evalnoise.decision_accuracy(np.array([1, 2, 3]), np.array([10, 20, 30])) == 1.0
    assert evalnoise.decision_accuracy(np.array([3, 2, 1]), np.array([10, 20, 30])) == 0.0
    last = np.array([[1.0, 1.2], [2.0, 2.2]])
    assert evalnoise.noise(last) == pytest.approx(0.1)
    assert evalnoise.snr(np.array([1.0, 3.0]), last) == pytest.approx(10.0)
    line = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    assert evalnoise.noise(line) > 1.0 and evalnoise.noise(line, detrend=True) < 1e-9
    assert evalnoise.noise(line + np.array([0, 0.1, -0.1, 0.1, 0]), detrend=True) > 0.05
    assert evalnoise.rank_flip_rate(np.array([1.0, 1.0, 3.0]), np.array([2.0, 2.0, 2.0])) == pytest.approx(1 / 3)
