"""Speculative decoding keeps the target's output: exactly at temperature 0, in distribution otherwise."""

import torch

from inferlab.decoder import SHAPES, Decoder
from inferlab.speculative import accept_or_resample, plain_generate, speculative_generate, toy_speculative_samples


def test_rejection_rule_reproduces_target_distribution():
    gen = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(6, generator=gen) * 2, 0)
    q = torch.softmax(torch.randn(6, generator=gen) * 2, 0)
    n = 200_000
    emp = torch.bincount(toy_speculative_samples(p, q, n, gen), minlength=6).float() / n
    assert 0.5 * (emp - p).abs().sum() < 0.01
    assert 0.5 * (q - p).abs().sum() > 0.1  # the draft really was different


def test_identical_draft_accepts_everything():
    p = torch.softmax(torch.randn(5, 10), -1)
    emitted, k = accept_or_resample(p, p[:4], torch.tensor([1, 2, 3, 4]), torch.Generator().manual_seed(0))
    assert k == 4 and emitted[:4] == [1, 2, 3, 4] and len(emitted) == 5


def test_greedy_speculative_equals_greedy_plain():
    target = Decoder.random(SHAPES["tiny"], 128, seed=1)
    draft = Decoder.random(SHAPES["tiny"], 128, seed=2)
    prompt = torch.randint(0, 256, (1, 9), generator=torch.Generator().manual_seed(0))
    base = plain_generate(target, prompt, 30)
    for gamma in (1, 3, 5):
        spec = speculative_generate(target, draft, prompt, 30, gamma=gamma)
        assert spec.tokens == base.tokens
        assert spec.target_passes < 30 or spec.alpha < 0.2


def test_self_draft_emits_gamma_plus_one_per_pass():
    target = Decoder.random(SHAPES["tiny"], 128, seed=1)
    draft = Decoder.random(SHAPES["tiny"], 128, seed=1)
    prompt = torch.randint(0, 256, (1, 5))
    st = speculative_generate(target, draft, prompt, 25, gamma=4)
    assert st.alpha == 1.0 and st.target_passes == 5 and len(st.tokens) == 25
