import math

import pytest
import torch
import torch.nn.functional as F

from stablelab.model import CharData, Config, TinyTransformer, deepnorm_constants, synthetic_corpus, train


@pytest.fixture(scope="module")
def data():
    return CharData(synthetic_corpus(8000), seq=16)


@pytest.mark.parametrize("norm,init", [("pre", "default"), ("pre", "scaled"), ("post", "default"),
                                       ("post", "xavier"), ("deep", "default")])
def test_forward_shapes(data, norm, init):
    model = TinyTransformer(Config(vocab=data.vocab, d=32, layers=2, heads=2, seq=16, norm=norm, init=init))
    x, y = data.batch(4, torch.Generator().manual_seed(0))
    assert model(x).shape == (4, 16, data.vocab)
    assert torch.isfinite(model.loss(x, y))


def test_deepnorm_constants_match_post():
    for m, a, b in ((12, 2.21, 0.32), (48, 3.13, 0.23), (500, 5.62, 0.13)):
        alpha, beta = deepnorm_constants(m)
        assert alpha == pytest.approx(a, abs=0.01)
        assert beta == pytest.approx(b, abs=0.01)


def test_deepnorm_scales_value_not_query(data):
    torch.manual_seed(0)
    model = TinyTransformer(Config(vocab=data.vocab, d=64, layers=12, heads=2, seq=16, norm="deep"))
    blk = model.blocks[0]
    xavier_std = math.sqrt(2 / 128)
    assert blk.q.weight.std().item() == pytest.approx(xavier_std, rel=0.1)
    assert blk.v.weight.std().item() == pytest.approx(xavier_std * deepnorm_constants(12)[1], rel=0.1)
    assert blk.alpha == pytest.approx(deepnorm_constants(12)[0])


def test_scaled_init_shrinks_residual_outputs(data):
    torch.manual_seed(0)
    model = TinyTransformer(Config(vocab=data.vocab, d=64, layers=8, heads=2, seq=16, init="scaled"))
    assert model.blocks[0].o.weight.std().item() == pytest.approx(0.02 / 4, rel=0.1)


def test_qk_norm_bounds_logits(data):
    torch.manual_seed(0)
    model = TinyTransformer(Config(vocab=data.vocab, d=32, layers=1, heads=2, seq=16, qk_norm=True))
    with torch.no_grad():
        model.blocks[0].q.weight.mul_(100)
        model.blocks[0].k.weight.mul_(100)
    model(data.batch(4, torch.Generator().manual_seed(0))[0])
    assert model.blocks[0].max_logit <= math.sqrt(16) + 1e-4


def test_softcap_bounds_logits(data):
    torch.manual_seed(0)
    model = TinyTransformer(Config(vocab=data.vocab, d=32, layers=1, heads=2, seq=16, softcap=5.0))
    with torch.no_grad():
        model.blocks[0].q.weight.mul_(1000)
    model(data.batch(4, torch.Generator().manual_seed(0))[0])
    assert model.blocks[0].max_logit <= 5.0


def test_layernorm_jacobian_scales_inverse_with_norm():
    u = torch.randn(64, dtype=torch.float64)
    u = (u - u.mean()) / (u - u.mean()).norm()
    for r in (1.0, 10.0):
        jac = torch.autograd.functional.jacobian(lambda z: F.layer_norm(z, (64,), eps=0.0), u * r)
        assert torch.linalg.matrix_norm(jac, ord=2).item() == pytest.approx(8.0 / r, rel=1e-6)


def test_training_reduces_loss(data):
    torch.manual_seed(0)
    model = TinyTransformer(Config(vocab=data.vocab, d=32, layers=2, heads=2, seq=16))
    hist = train(model, data, steps=60, lr=3e-3, warmup=10, batch=16, log_every=20)
    assert not hist["diverged"]
    assert hist["loss"][-1] < hist["loss"][0] - 0.5
    assert len(hist["max_logit"]) == len(hist["step"])
