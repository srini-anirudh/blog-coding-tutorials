import pytest
import torch

from stablelab.mup import MLP, context_dataset, coordinate_check
from stablelab.numerics import DynamicLossScaler, format_table, fraction_lost
from stablelab.optim import InspectableAdam, clip_global_norm, stress_stream


def test_inspectable_adam_matches_torch():
    p = torch.zeros(32, dtype=torch.float64, requires_grad=True)
    ref = torch.optim.Adam([p], lr=1e-2, betas=(0.9, 0.99), eps=1e-8)
    ours = InspectableAdam(32, lr=1e-2, betas=(0.9, 0.99))
    total = torch.zeros(32, dtype=torch.float64)
    for _ in range(50):
        g = torch.randn(32, dtype=torch.float64)
        p.grad = g.clone()
        ref.step()
        total += ours.step(g)["update"]
    assert torch.allclose(p.detach(), total, atol=1e-12)


def test_first_adam_step_is_sign():
    g = torch.randn(100, dtype=torch.float64) * 1e-5
    up = InspectableAdam(100, lr=1.0, eps=1e-12).step(g)["update"]  # sign(g) exactly when eps << |g|
    assert torch.allclose(up, -torch.sign(g), atol=1e-2)


def test_stale_second_moment_overshoots():
    opt = InspectableAdam(1, lr=1.0, betas=(0.9, 0.999))
    for _ in range(2000):
        opt.step(torch.zeros(1, dtype=torch.float64))
    peak = max(opt.step(torch.ones(1, dtype=torch.float64))["update"].abs().item() for _ in range(50))
    assert peak > 5.0


def test_stress_stream_phases():
    g = stress_stream(steps=1600, n=64)
    assert g[:20].std() > 5 and g[500:1400].std() < 0.05 and g[1500:1559].mean() > 0.8


def test_global_clip_keeps_direction():
    a, b = torch.randn(10) * 10, torch.randn(5) * 10
    (ca, cb), norm = clip_global_norm([a, b], 1.0)
    full, clipped = torch.cat([a, b]), torch.cat([ca, cb])
    assert clipped.norm().item() == pytest.approx(1.0, rel=1e-5)
    assert torch.allclose(clipped / clipped.norm(), full / full.norm(), atol=1e-6)


def test_format_table_limits():
    rows = {r["format"]: r for r in format_table()}
    assert rows["FP16"]["max"] == 65504
    assert rows["FP16"]["mant"] == 10 and rows["BF16"]["mant"] == 7
    assert rows["FP8 E4M3"]["max"] == 448
    assert rows["BF16"]["exp"] == rows["FP32"]["exp"] == 8


def test_fraction_lost_and_scaling():
    v = torch.full((100,), 1e-9)
    assert fraction_lost(v, torch.float16)["zero"] == 1.0
    assert fraction_lost(v, torch.float16, scale=2.0**24)["zero"] == 0.0
    assert fraction_lost(torch.full((10,), 1e3), torch.float16, scale=2.0**8)["inf"] == 1.0


def test_dynamic_loss_scaler():
    s = DynamicLossScaler(init_scale=1024.0, growth_interval=3)
    assert s.update(True) is False and s.scale == 512.0
    for _ in range(3):
        assert s.update(False)
    assert s.scale == 1024.0 and s.skipped == 1


def test_mup_equals_sp_at_base_width():
    torch.manual_seed(0)
    sp = MLP(10, 64, 5, "sp")
    torch.manual_seed(0)
    mu = MLP(10, 64, 5, "mup")
    assert torch.equal(sp.out.weight, mu.out.weight)
    assert [g["lr"] for g in mu.param_groups(1.0)] == [1.0, 1.0]


def test_mup_scales_hidden_lr_and_readout():
    torch.manual_seed(0)
    mu = MLP(10, 256, 5, "mup")
    assert [g["lr"] for g in mu.param_groups(1.0)] == [1.0, 0.25]
    assert mu.out.weight.std().item() == pytest.approx((1 / 16) * 0.5, rel=0.15)


def test_coordinate_check_runs():
    ids = torch.randint(0, 7, (3000,))
    x, y = context_dataset(ids, 7, context=4)
    assert x.shape == (2996, 28)
    out = coordinate_check(64, "mup", x, y, steps=2)
    assert set(out) == {"h1", "h2", "out"} and all(v > 0 for v in out.values())
