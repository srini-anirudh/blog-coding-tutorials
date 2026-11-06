"""Paths that need a real CUDA device. Run with: python -m pytest c01-pytorch -m gpu"""

import pytest
import torch

from scratchnet import optim as manual
from scratchnet.data import SimpleDataLoader, SpiralDataset
from scratchnet.debug import check_same_device
from scratchnet.device import pick_device, synchronize
from scratchnet.layers import MLP
from scratchnet.losses import cross_entropy
from scratchnet.train import TrainConfig, fit

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def test_cuda_and_to_are_the_same_copy():
    x = torch.randn(4, 8)
    a, b = x.cuda(), x.to("cuda")
    assert pick_device().type == "cuda" and a.device == b.device
    assert x.device.type == "cpu"  # .to returns a new tensor; the source stays where it was
    assert torch.equal(a, b) and torch.equal(a.cpu(), x)


def test_pinned_non_blocking_copy_arrives_after_sync():
    x = torch.randn(1024).pin_memory()
    assert x.is_pinned()
    y = x.to("cuda", non_blocking=True)
    synchronize("cuda")
    assert torch.equal(y.cpu(), x)


def test_real_device_mismatch_is_caught():
    model = MLP(2, 8, 3).cuda()
    x = torch.randn(4, 2)
    with pytest.raises(RuntimeError, match="same device"):
        model(x)
    with pytest.raises(RuntimeError, match="x is on cpu"):
        check_same_device(model, x=x)


def test_manual_adamw_matches_torch_on_cuda():
    torch.manual_seed(0)
    a = MLP(2, 16, 3).cuda()
    b = MLP(2, 16, 3).cuda()
    b.load_state_dict(a.state_dict())
    opt_a, opt_b = manual.AdamW(a.parameters(), lr=1e-2), torch.optim.AdamW(b.parameters(), lr=1e-2)
    x, y = torch.randn(32, 2, device="cuda"), torch.randint(0, 3, (32,), device="cuda")
    for model, opt in ((a, opt_a), (b, opt_b)):
        for _ in range(10):
            opt.zero_grad()
            cross_entropy(model(x), y).backward()
            opt.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-6)


def test_resume_on_cuda(tmp_path):
    base = dict(optimizer="manual-adamw", lr=1e-2, points_per_class=60, device="cuda")
    _, full = fit(TrainConfig(epochs=4, **base))
    fit(TrainConfig(epochs=2, **base), save_to=tmp_path / "ckpt.pt")
    _, resumed = fit(TrainConfig(epochs=4, **base), resume_from=tmp_path / "ckpt.pt")
    assert resumed[-1]["train_loss"] == pytest.approx(full[-1]["train_loss"], rel=1e-5)


def test_simple_loader_pins_batches():
    batch = next(iter(SimpleDataLoader(SpiralDataset(10), batch_size=4, pin_memory=True)))
    assert batch["x"].is_pinned() and batch["y"].is_pinned()
