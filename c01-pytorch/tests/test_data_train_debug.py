import pytest
import torch
from torch.utils.data import DataLoader

from scratchnet.data import SpiralDataset, VarLenDataset, collate_dicts, pad_collate
from scratchnet.debug import ActivationStats, GradNormMonitor, NaNDetector, check_same_device
from scratchnet.layers import MLP, Linear, ReLU
from scratchnet.losses import cross_entropy
from scratchnet.train import TrainConfig, fit


def test_collate_dicts_stacks():
    ds = SpiralDataset(10)
    batch = next(iter(DataLoader(ds, batch_size=8, collate_fn=collate_dicts)))
    assert batch["x"].shape == (8, 2) and batch["y"].shape == (8,) and batch["y"].dtype == torch.long


def test_pad_collate_shapes_and_mask():
    ds = VarLenDataset(n=16)
    loader = DataLoader(ds, batch_size=4, collate_fn=pad_collate)
    batch = next(iter(loader))
    t_max = int(batch["lengths"].max())
    assert batch["input_ids"].shape == (4, t_max) and batch["attention_mask"].shape == (4, t_max)
    assert torch.equal(batch["attention_mask"].sum(1), batch["lengths"])
    assert (batch["input_ids"][batch["attention_mask"] == 0] == 0).all()
    with pytest.raises(RuntimeError):
        next(iter(DataLoader(ds, batch_size=4)))  # default collate cannot stack ragged tensors


def test_training_learns_spirals():
    _, history = fit(TrainConfig(optimizer="adamw", lr=1e-2, epochs=40, points_per_class=200))
    assert history[-1]["val_acc"] > 0.9
    assert history[-1]["train_loss"] < history[0]["train_loss"] / 3


def test_resume_matches_uninterrupted_run(tmp_path):
    base = dict(optimizer="manual-adamw", lr=1e-2, points_per_class=60)
    _, full = fit(TrainConfig(epochs=4, **base))
    fit(TrainConfig(epochs=2, **base), save_to=tmp_path / "ckpt.pt")
    _, resumed = fit(TrainConfig(epochs=4, **base), resume_from=tmp_path / "ckpt.pt")
    assert [r["epoch"] for r in resumed] == [2, 3]
    assert resumed[-1]["train_loss"] == pytest.approx(full[-1]["train_loss"], rel=1e-6)


def test_grad_monitor_sees_reverse_layer_order():
    model = MLP(2, 8, 3, depth=2)
    with GradNormMonitor(model) as mon:
        cross_entropy(model(torch.randn(5, 2)), torch.tensor([0, 1, 2, 0, 1])).backward()
    first_layer = [n for n in mon.record.order if n.startswith("blocks.0")]
    assert mon.record.order.index("head.weight") < mon.record.order.index(first_layer[0])
    assert set(mon.record.norms) == {n for n, _ in model.named_parameters()}
    assert not model.blocks[0][0].weight._post_accumulate_grad_hooks  # removed on exit


def test_activation_stats_and_nan_detector():
    model = MLP(2, 8, 3)
    with ActivationStats(model, (Linear, ReLU)) as acts:
        model(torch.randn(64, 2))
    assert 0.0 < acts.stats["blocks.0.1"].frac_zero < 1.0
    with torch.no_grad():
        model.blocks[1][0].weight[0, 0] = float("inf")
    with NaNDetector(model), pytest.raises(FloatingPointError, match="first appears in blocks.1.0"):
        model(torch.randn(4, 2))


def test_device_check_names_the_tensor():
    model = MLP(2, 4, 3)
    check_same_device(model, x=torch.randn(2, 2))
    if torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="x is on mps"):
            check_same_device(model, x=torch.randn(2, 2, device="mps"))
