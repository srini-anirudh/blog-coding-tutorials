import subprocess
import sys
from pathlib import Path

import pytest
import torch

from twomod import BlockConfig, ViT
from twomod.train import TrainConfig, augment, fit, lr_at

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_augment_keeps_shape_and_pixels():
    x = torch.randint(1, 255, (8, 3, 32, 32), dtype=torch.uint8)
    out = augment(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    kept = out[out != 0]  # crops only move pixels or pad with zeros; flips only move them
    assert set(kept.unique().tolist()) <= set(x.unique().tolist())


def test_lr_schedule():
    cfg = TrainConfig(steps=100, lr=1.0, warmup_frac=0.1)
    assert lr_at(0, cfg) == pytest.approx(0.1) and lr_at(9, cfg) == pytest.approx(1.0)
    assert lr_at(99, cfg) < 0.01


def test_fit_learns_a_trivial_task():
    torch.manual_seed(0)
    x = torch.zeros(64, 3, 16, 16, dtype=torch.uint8)
    y = torch.arange(64) % 2
    x[y == 1] = 255  # class 1 is white, class 0 is black
    model = ViT(BlockConfig(dim=32, n_heads=4), depth=1, image_size=16, patch=4, num_classes=2)
    out = fit(model, x, y, x, y, TrainConfig(steps=40, batch_size=16, lr=3e-3))
    assert out["test_acc"] == 1.0


@pytest.mark.parametrize("script", ["03_position.py", "04_masks.py", "05_identical_middle.py",
                                    "06_inductive_bias.py"])
def test_offline_script_runs(script):
    out = subprocess.run([sys.executable, str(SCRIPTS / script)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("# hardware:")


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("script", ["01_front_doors.py", "02_sequences.py"])
def test_network_script_runs(script):
    out = subprocess.run([sys.executable, str(SCRIPTS / script)], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr


@pytest.mark.network
@pytest.mark.slow
def test_training_script_quick_mode():
    out = subprocess.run([sys.executable, str(SCRIPTS / "07_vit_vs_cnn.py"), "--quick"],
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr
    assert "vit-learned" in out.stdout and "cnn" in out.stdout
