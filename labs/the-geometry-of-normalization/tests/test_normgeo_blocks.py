"""Placements in the tiny transformer, and a smoke run of every script."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from normgeo.blocks import PLACEMENTS, TinyLM, lm_loss

LAB = Path(__file__).resolve().parents[1]


def stream_rms(placement: str, layers: int = 16) -> list[float]:
    torch.manual_seed(0)
    model = TinyLM(32, 32, layers, 4, placement, 16)
    with torch.no_grad():
        _, stream = model(torch.randint(0, 32, (4, 16)))
    return [s.pow(2).mean().sqrt().item() for s in stream]


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_every_placement_runs_and_trains(placement):
    torch.manual_seed(0)
    model = TinyLM(32, 32, 4, 4, placement, 16)
    x = torch.randint(0, 32, (8, 16))
    y = torch.roll(x, -1, 1)
    logits, stream = model(x)
    assert logits.shape == (8, 16, 32) and len(stream) == 5
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = lm_loss(model, x, y).item()
    for _ in range(30):
        loss = lm_loss(model, x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_depth_profiles_at_init():
    post, pre, lns = stream_rms("post"), stream_rms("pre"), stream_rms("lns")
    assert all(abs(r - 1) < 0.05 for r in post[1:])  # the norm on the highway pins the stream
    assert pre[-1] > 1.5 * pre[4]  # Pre-LN: variance accumulates with depth
    assert lns[-1] < pre[-1]  # 1/sqrt(l) damps the accumulation


def test_skipping_a_block_changes_the_output():
    torch.manual_seed(0)
    model = TinyLM(32, 32, 4, 4, "pre", 16)
    x = torch.randint(0, 32, (2, 16))
    with torch.no_grad():
        a, _ = model(x)
        model.skip = {2}
        b, _ = model(x)
    assert (a - b).abs().max() > 1e-4


FAST = [
    ("01_axis_atlas.py", []),
    ("02_anatomy.py", ["--reps", "2"]),
    ("03_effective_lr.py", ["--steps", "200"]),
    ("04_batchnorm.py", []),
    ("06_depth.py", ["--layers", "4", "--steps", "5", "--placements", "pre", "post"]),
    ("07_qk_norm.py", []),
    ("08_llm_nuances.py", []),
    ("09_interface.py", []),
]


def run(script: str, args: list[str]) -> str:
    env = dict(os.environ, NORMGEO_NO_WRITE="1", OMP_NUM_THREADS="2")
    out = subprocess.run([sys.executable, str(LAB / "scripts" / script), *args], env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


@pytest.mark.parametrize("script,args", FAST)
def test_script_runs(script, args):
    assert "# " in run(script, args).splitlines()[0]  # starts with the hardware line


@pytest.mark.slow
@pytest.mark.network
def test_small_batch_script():
    run("05_small_batch.py", ["--train", "512", "--test", "256", "--epochs", "1", "--batches", "8", "64"])


@pytest.mark.slow
@pytest.mark.network
def test_real_models_script():
    out = run("10_real_models.py", ["--windows", "1", "--seq", "64", "--images", "2"])
    assert "after the connector" in out
