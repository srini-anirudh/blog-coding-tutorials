"""Sampling filters: what each one keeps and what it removes."""

import torch

from tinyllm.sampling import apply_top_k, apply_top_p, sample_next_token


def test_greedy_is_argmax():
    logits = torch.tensor([[0.1, 3.0, 2.0], [5.0, 0.0, 1.0]])
    assert sample_next_token(logits, temperature=0.0).tolist() == [[1], [0]]


def test_top_k_keeps_k_largest():
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    kept = torch.isfinite(apply_top_k(logits, 2))
    assert kept.tolist() == [[False, True, True, False]]


def test_top_p_keeps_smallest_nucleus():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    assert torch.isfinite(apply_top_p(logits, 0.75)).tolist() == [[True, True, False, False]]
    assert torch.isfinite(apply_top_p(logits, 0.8)).tolist() == [[True, True, False, False]]
    assert torch.isfinite(apply_top_p(logits, 0.81)).tolist() == [[True, True, True, False]]
    # Even a tiny p keeps the top token.
    assert torch.isfinite(apply_top_p(logits, 0.01)).tolist() == [[True, False, False, False]]


def test_top_p_works_on_unsorted_logits():
    logits = torch.tensor([[0.15, 0.5, 0.05, 0.3]]).log()
    assert torch.isfinite(apply_top_p(logits, 0.75)).tolist() == [[False, True, False, True]]


def test_low_temperature_concentrates_samples():
    logits = torch.tensor([[2.0, 1.0, 0.0]]).expand(2000, 3)
    g = torch.Generator().manual_seed(0)
    cold = sample_next_token(logits, temperature=0.1, generator=g)
    hot = sample_next_token(logits, temperature=10.0, generator=g)
    assert (cold == 0).float().mean() > 0.99
    assert 0.25 < (hot == 0).float().mean() < 0.45


def test_sampling_is_reproducible_with_seed():
    logits = torch.randn(4, 50)
    a = sample_next_token(logits, 1.0, generator=torch.Generator().manual_seed(7))
    b = sample_next_token(logits, 1.0, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
