"""The operators, their invariances and their hand-derived gradients."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from normgeo.geometry import GradScale, rope, rotation_angle
from normgeo.ops import DyT, RMSNorm, atlas, layernorm_backward, rmsnorm_backward, weight_decay_groups

torch.manual_seed(0)


def test_atlas_matches_torch():
    x = torch.randn(4, 6, 5) * 2 + 1
    ref = {
        "batchnorm": F.batch_norm(x, None, None, training=True),
        "layernorm": F.layer_norm(x.transpose(1, 2), (6,)).transpose(1, 2),
        "instancenorm": F.instance_norm(x),
        "groupnorm": F.group_norm(x, 2),
        "rmsnorm": F.rms_norm(x.transpose(1, 2), (6,), eps=1e-5).transpose(1, 2),
    }
    for kind, r in ref.items():
        torch.testing.assert_close(atlas(x, kind, groups=2), r, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", ["batchnorm", "layernorm", "instancenorm", "groupnorm", "rmsnorm"])
def test_only_batchnorm_mixes_samples(kind):
    x = torch.randn(4, 6, 5)
    x2 = x.clone()
    x2[1] = x2[1] * 10 + 5
    moved = (atlas(x2, kind)[0] - atlas(x, kind)[0]).abs().max().item()
    assert (moved > 0.1) == (kind == "batchnorm")


def test_invariances():
    x = torch.randn(32, 64, dtype=torch.float64)
    ln = lambda t: F.layer_norm(t, (64,), eps=1e-12)  # noqa: E731
    rms = lambda t: F.rms_norm(t, (64,), eps=1e-12)  # noqa: E731
    torch.testing.assert_close(ln(5 * x), ln(x))
    torch.testing.assert_close(rms(5 * x), rms(x))
    torch.testing.assert_close(ln(x + 3), ln(x))  # re-centring: LayerNorm only
    assert (rms(x + 3) - rms(x)).abs().max() > 0.5


def test_backward_formulas_match_autograd_and_project():
    x = torch.randn(16, 32, dtype=torch.float64) * 2 + 1
    gamma, dy = torch.randn(32, dtype=torch.float64), torch.randn(16, 32, dtype=torch.float64)
    for fn, back in [(lambda t: F.layer_norm(t, (32,), eps=1e-5), layernorm_backward),
                     (lambda t: F.rms_norm(t, (32,), eps=1e-5), rmsnorm_backward)]:
        xr = x.clone().requires_grad_(True)
        (fn(xr) * gamma * dy).sum().backward()
        torch.testing.assert_close(back(x, gamma, dy), xr.grad)
    g_ln = layernorm_backward(x, gamma, dy)
    assert g_ln.sum(-1).abs().max() < 1e-10  # orthogonal to the all-ones vector
    assert rmsnorm_backward(x, gamma, dy).sum(-1).abs().max() > 1e-3  # RMSNorm keeps that component


def test_rmsnorm_conventions():
    x = torch.randn(8, 128)
    torch.testing.assert_close(RMSNorm(128)(x), F.rms_norm(x, (128,), eps=1e-6))
    gemma = RMSNorm(128, offset=True)
    torch.testing.assert_close(gemma(x), RMSNorm(128)(x))  # (1 + 0) = 1 at init
    small = torch.randn(8, 128) * 1e-3
    a, b = RMSNorm(128)(small), RMSNorm(128, eps_inside=False)(small)
    assert ((a - b).abs() / b.abs()).max() > 0.1  # the two eps conventions disagree near sqrt(eps)
    xb = (torch.randn(4, 4096) * 100).bfloat16()
    assert RMSNorm(4096)(xb).dtype == torch.bfloat16


def test_weight_decay_groups_skip_norms_biases_embeddings():
    model = nn.Sequential(nn.Embedding(10, 8), nn.Linear(8, 8), nn.LayerNorm(8), RMSNorm(8))
    decay, no_decay = weight_decay_groups(model, 0.1)
    assert [p.shape for p in decay["params"]] == [torch.Size([8, 8])]
    assert len(no_decay["params"]) == 5 and no_decay["weight_decay"] == 0.0


def test_dyt_is_pointwise():
    dyt = DyT(16)
    x = torch.randn(4, 16)
    x2 = x.clone()
    x2[:, 1:] *= 100
    torch.testing.assert_close(dyt(x)[:, 0], dyt(x2)[:, 0])  # no statistics across elements


def test_rope_keeps_norm_and_commutes_with_unit_gain_norm():
    q = torch.randn(2, 32, 16)
    torch.testing.assert_close(rope(q).norm(dim=-1), q.norm(dim=-1))
    n = lambda t: F.rms_norm(t, (16,))  # noqa: E731
    torch.testing.assert_close(n(rope(q)), rope(n(q)), atol=1e-5, rtol=1e-5)


def test_rotation_angle_formula():
    h = torch.zeros(8)
    h[0] = 30.0
    u = torch.zeros(8)
    u[1] = 1.0
    measured = math.acos(F.cosine_similarity(h, h + u, dim=0).item())
    assert abs(measured - rotation_angle(30.0, 1.0, math.pi / 2)) < 1e-6


def test_grad_scale_hook():
    x = torch.randn(4, 3, requires_grad=True)
    y = GradScale.apply(x, 10.0)
    torch.testing.assert_close(y, x)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 10.0))
