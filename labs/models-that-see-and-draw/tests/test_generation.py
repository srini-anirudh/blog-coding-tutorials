"""Data, rectified flow, and the regression-to-the-mean claim."""

import torch
import torch.nn.functional as F

from seedraw import data, flow


def test_cat_images_have_two_modes():
    x = data.cat_images(200, torch.Generator().manual_seed(0))
    bal = data.side_balance(x)
    assert set(bal.round().tolist()) == {0.0, 1.0}
    assert len(torch.unique(x.flatten(1), dim=0)) == 6


def test_shapes_and_patches_round_trip():
    d = data.shapes(32, seed=0)
    assert d["images"].shape == (32, 1, 16, 16)
    assert 0 <= d["images"].min() and d["images"].max() <= 1
    p = data.to_patches(d["images"])
    assert p.shape == (32, 16, 16)
    assert torch.equal(data.from_patches(p), d["images"])
    # the first patch is the top-left 4 x 4 block
    assert torch.equal(p[0, 0].view(4, 4), d["images"][0, 0, :4, :4])


def test_squared_error_optimum_is_the_mean_image():
    x = data.cat_images(512, torch.Generator().manual_seed(1))
    f = torch.zeros(8, 8, requires_grad=True)
    opt = torch.optim.SGD([f], lr=16.0)  # the per-pixel gradient is 2 (f - mean) / 64
    for _ in range(200):
        loss = F.mse_loss(f.expand_as(x), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.allclose(f.detach(), x.mean(0), atol=1e-4)
    assert f.max() < 0.6  # no pixel is confidently "cat"


def test_noisy_point_endpoints_and_target():
    x0, eps = torch.randn(4, 3), torch.randn(4, 3)
    assert torch.allclose(flow.noisy_point(x0, torch.zeros(4), eps), x0)
    assert torch.allclose(flow.noisy_point(x0, torch.ones(4), eps), eps)
    # moving along -target from x_t by t lands exactly on x0 (the line is straight)
    t = torch.full((4,), 0.3)
    xt = flow.noisy_point(x0, t, eps)
    assert torch.allclose(xt - 0.3 * flow.flow_target(x0, eps), x0, atol=1e-6)


def test_euler_sampler_with_exact_velocity_reaches_the_data_point():
    target = torch.tensor([[2.0, -1.0]])

    class Exact(torch.nn.Module):
        def forward(self, xt, t, c):  # for one data point x0, v = (x_t - x0) / t
            return (xt - target) / t[:, None]

    out = flow.sample(Exact(), (5, 2), torch.zeros(5, dtype=torch.long), steps=10)
    assert torch.allclose(out, target.expand(5, 2), atol=1e-5)


def test_flow_loss_backpropagates():
    v = flow.VelocityMLP(64)
    x = data.cat_images(16).flatten(1)
    loss = flow.flow_loss(v, x, torch.zeros(16, dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss) and v.net[0].weight.grad.abs().sum() > 0
