"""The tiny model, the synthetic tasks, and the symmetry facts of sections 01 and 07."""

import pytest
import torch

from posenc import extend, tasks, tinylm


def _gap(attn, x, causal):
    perm = torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        return float((attn(x[:, perm], causal=causal) - attn(x, causal=causal)[:, perm]).abs().max())


def test_attention_without_position_is_permutation_equivariant():
    torch.manual_seed(0)
    attn, x = tinylm.Attention(16, 2), torch.randn(1, 9, 16)
    assert _gap(attn, x, causal=False) < 1e-5
    assert _gap(attn, x, causal=True) > 1e-2


@pytest.mark.parametrize("pe", tinylm.PE_KINDS)
def test_every_scheme_runs_and_is_causal(pe):
    torch.manual_seed(0)
    m = tinylm.TinyLM(20, d=32, heads=2, pe=pe, max_len=16).eval()
    t = torch.randint(20, (2, 16))
    out = m(t)
    assert out.shape == (2, 16, 20)
    t2 = t.clone()
    t2[:, -1] = (t2[:, -1] + 1) % 20  # changing the last token must not change earlier outputs
    torch.testing.assert_close(m(t2)[:, :-1], out[:, :-1])


def test_learned_table_has_a_hard_limit():
    m = tinylm.TinyLM(10, d=16, heads=2, pe="learned", max_len=8)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 9, dtype=torch.long))


def test_rope_override_with_unscaled_freqs_is_identity():
    torch.manual_seed(0)
    m = tinylm.TinyLM(10, d=32, heads=2, pe="rope").eval()
    t = torch.randint(10, (1, 12))
    f, s = extend.scaled_inv_freq("none", m.head_dim, 10_000, 1, 64)
    torch.testing.assert_close(m(t, rope_freqs=f, logit_scale=s), m(t))


def test_induction_batch_masks_only_copied_tokens():
    gen = torch.Generator().manual_seed(0)
    tokens, targets, mask = tasks.induction_batch(gen, batch=16, length=64, seg=12)
    assert tokens.shape == targets.shape == mask.shape == (16, 64)
    assert (mask.sum(1) == 11).all()
    torch.testing.assert_close(tokens[:, 1:], targets[:, :-1])
    for b in range(16):
        t = int(mask[b].nonzero()[0])
        # the current token at the start of the copy also occurs earlier, followed by the target
        earlier = (tokens[b, :t] == tokens[b, t]).nonzero().flatten()
        assert any(int(targets[b, e]) == int(targets[b, t]) for e in earlier)


def test_training_reduces_loss():
    torch.manual_seed(0)
    m = tinylm.TinyLM(64, n_out=16, d=32, heads=2, pe="none", max_len=16)
    losses = tinylm.train(m, tasks.make("position", length=16, vocab=64, batch=32), 60, lr=3e-3)
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10
