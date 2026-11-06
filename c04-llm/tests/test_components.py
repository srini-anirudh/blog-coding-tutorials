"""Unit tests for the pieces: RMSNorm, RoPE, attention backends, SwiGLU, loss."""

import math

import pytest
import torch
import torch.nn.functional as F

from c04_helpers import random_ids, small_config, small_model
from tinyllm.attention_naive import causal_mask, naive_attention, repeat_kv
from tinyllm.attention_sdpa import sdpa_attention
from tinyllm.loss import causal_lm_loss
from tinyllm.mlp import SwiGLU
from tinyllm.norm import RMSNorm
from tinyllm.rope import RotaryEmbedding, rotate


def test_rmsnorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 64) * 7
    ours = RMSNorm(64, eps=1e-5)
    ours.weight.data = torch.rand(64) + 0.5
    ref = torch.nn.RMSNorm(64, eps=1e-5)
    ref.weight.data = ours.weight.data.clone()
    torch.testing.assert_close(ours(x), ref(x))


def test_rmsnorm_output_has_unit_rms():
    x = torch.randn(4, 10, 64) * 100
    y = RMSNorm(64)(x)
    torch.testing.assert_close(y.pow(2).mean(-1).sqrt(), torch.ones(4, 10), atol=1e-4, rtol=0)


def test_rope_preserves_norm():
    rope = RotaryEmbedding(16, max_seq_len=32)
    x = torch.randn(2, 3, 9, 16)
    cos, sin = rope(torch.arange(9).expand(2, 9))
    torch.testing.assert_close(rotate(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


def test_rope_dot_product_depends_only_on_offset():
    """q at position m and k at position n: the score depends on m - n, not on m and n."""
    torch.manual_seed(0)
    rope = RotaryEmbedding(16, max_seq_len=128)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m: int, n: int) -> torch.Tensor:
        cm, sm = rope(torch.tensor([[m]]))
        cn, sn = rope(torch.tensor([[n]]))
        return (rotate(q, cm, sm) * rotate(k, cn, sn)).sum()

    torch.testing.assert_close(score(5, 2), score(103, 100), atol=1e-4, rtol=0)
    assert not torch.allclose(score(5, 2), score(5, 3))


def test_repeat_kv_groups_heads():
    k = torch.arange(2).float().view(1, 2, 1, 1).expand(1, 2, 3, 4)  # kv head 0 is all 0, head 1 all 1
    r = repeat_kv(k, 3)
    assert r.shape == (1, 6, 3, 4)
    assert r[0, :, 0, 0].tolist() == [0, 0, 0, 1, 1, 1]


def test_causal_mask_shifts_with_cache():
    assert causal_mask(3, 3, torch.device("cpu")).int().tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    # Two new queries after three cached positions: positions 3 and 4.
    assert causal_mask(2, 5, torch.device("cpu")).int().tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]


@pytest.mark.parametrize("q_len,kv_len", [(7, 7), (1, 7), (3, 7)])
@pytest.mark.parametrize("n_rep", [1, 2, 4])
def test_naive_and_sdpa_agree(q_len, kv_len, n_rep):
    torch.manual_seed(0)
    h = 4
    q = torch.randn(2, h, q_len, 8)
    k, v = torch.randn(2, h // n_rep, kv_len, 8), torch.randn(2, h // n_rep, kv_len, 8)
    torch.testing.assert_close(sdpa_attention(q, k, v, n_rep), naive_attention(q, k, v, n_rep), atol=1e-5, rtol=1e-5)


def test_naive_matches_sdpa_reference_mha():
    """Against PyTorch itself, not just against our own SDPA wrapper."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(naive_attention(q, k, v, 1), ref, atol=1e-5, rtol=1e-5)


def test_swiglu_shapes_and_formula():
    torch.manual_seed(0)
    mlp = SwiGLU(16, 40)
    x = torch.randn(2, 3, 16)
    expected = mlp.down_proj(F.silu(x @ mlp.gate_proj.weight.T) * (x @ mlp.up_proj.weight.T))
    torch.testing.assert_close(mlp(x), expected)
    assert mlp(x).shape == x.shape


def test_loss_is_shifted():
    """Position t is scored against token t+1, never against itself."""
    logits = torch.full((1, 4, 10), -1e4)
    ids = torch.tensor([[1, 2, 3, 4]])
    for t in range(3):
        logits[0, t, ids[0, t + 1]] = 0.0  # perfect next-token prediction
    assert causal_lm_loss(logits, ids).item() < 1e-6
    unshifted = F.cross_entropy(logits[0], ids[0])
    assert unshifted.item() > 100


def test_loss_ignores_masked_labels():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 10)
    labels = torch.tensor([[1, 2, -100, -100, 5]])
    # Only position 3 (predicting label 5) and position 0 (predicting label 2) count.
    expected = F.cross_entropy(logits[0, [0, 3]], torch.tensor([2, 5]))
    torch.testing.assert_close(causal_lm_loss(logits, labels), expected)


def test_initial_loss_is_near_uniform():
    """A freshly initialised model should be close to guessing: loss about ln(V)."""
    model = small_model(vocab_size=97)
    ids = random_ids(4, 32)
    loss = model(ids, labels=ids).loss.item()
    assert abs(loss - math.log(97)) < 0.15


def test_gradients_reach_every_parameter():
    model = small_model().train()
    ids = random_ids(2, 16)
    model(ids, labels=ids).loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_config_rejects_bad_heads():
    with pytest.raises(ValueError):
        small_config(num_heads=4, num_kv_heads=3)
