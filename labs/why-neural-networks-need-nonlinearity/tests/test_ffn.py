import pytest
import torch

from nonlin.data import random_windows, synthetic_bytes
from nonlin.ffn import FFN, FFN_KINDS, ffn_params, matched_hidden, round_up, sparse_down


def test_8d3_rule_gives_llama2_width():
    assert round_up(8 * 4096 / 3, 256) == 11008
    assert matched_hidden(4096, "swiglu", multiple_of=256) == 11008
    assert matched_hidden(4096, "relu") == 4 * 4096


@pytest.mark.parametrize("kind", sorted(FFN_KINDS))
def test_every_kind_runs_and_budget_matches(kind):
    d = 96
    h = matched_hidden(d, kind)
    m = FFN(d, h, kind, record=True)
    x = torch.randn(2, 5, d)
    assert m(x).shape == x.shape
    assert m.last_hidden.shape == (2, 5, h)
    n = sum(p.numel() for p in m.parameters())
    assert n == ffn_params(d, h, FFN_KINDS[kind][1])
    assert abs(n - 2 * d * 4 * d) / (2 * d * 4 * d) < 0.01


def test_unknown_kind():
    with pytest.raises(ValueError):
        FFN(4, 4, "nope")


def test_bilinear_represents_sum_of_products_exactly():
    m = FFN(8, 4, "bilinear")
    with torch.no_grad():
        m.gate_proj.weight.zero_()
        m.up_proj.weight.zero_()
        m.down_proj.weight.zero_()
        for k in range(4):
            m.gate_proj.weight[k, 2 * k] = 1.0
            m.up_proj.weight[k, 2 * k + 1] = 1.0
            m.down_proj.weight[0, k] = 1.0
    x = torch.rand(100, 8) * 2 - 1
    target = (x[:, 0::2] * x[:, 1::2]).sum(1)
    assert torch.allclose(m(x)[:, 0], target, atol=1e-6)


def test_relu2_equals_tied_reglu():
    """Squared ReLU is ReGLU with the gate and value projections tied."""
    a = FFN(16, 32, "relu2")
    b = FFN(16, 32, "reglu")
    with torch.no_grad():
        b.gate_proj.weight.copy_(a.up_proj.weight)
        b.up_proj.weight.copy_(a.up_proj.weight)
        b.down_proj.weight.copy_(a.down_proj.weight)
    x = torch.randn(4, 16)
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_sparse_down_equals_dense():
    torch.manual_seed(0)
    w_t = torch.randn(300, 64)
    a = torch.relu(torch.randn(300) - 1.0)
    out, rows = sparse_down(a, w_t)
    assert torch.allclose(out, a @ w_t, atol=1e-5)
    assert rows == int((a != 0).sum())
    _, rows_gelu = sparse_down(torch.nn.functional.gelu(torch.randn(300)), w_t)
    assert rows_gelu == 300


@pytest.mark.parametrize("kind", ["relu", "swiglu", "bilinear"])
def test_ffn_swapped_into_c4_tinyllm_trains(script, kind):
    mod = script("05_ffn_variants")
    torch.manual_seed(0)
    model, h = mod.build_lm(kind, d=32, layers=2, seq=32)
    assert all(isinstance(b.mlp, FFN) and b.mlp.kind == kind for b in model.layers)
    data = synthetic_bytes(20_000)
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        ids = random_windows(data, 8, 32, gen)
        loss = model(ids, labels=ids).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-5:]) / 5 < losses[0] - 0.5


def test_train_lm_job_on_synthetic(script):
    mod = script("05_ffn_variants")
    r = mod.train_lm(dict(kind="geglu", seed=0, d=32, layers=1, seq=32, batch=4, lr=3e-3, steps=5,
                          val_batches=2, threads=1, synthetic=True))
    assert r["h"] == matched_hidden(32, "geglu", multiple_of=8)
    assert r["val_loss"] > 0 and r["down_input_stats"]["exact_zero"] == 0.0
