"""Run the scripts end to end. The fast ones run offline; the dataset and model ones are marked."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(name: str, *args: str, tmp_cwd: Path) -> str:
    env = {**os.environ, "OMP_NUM_THREADS": "2", "VITFAMILY_RESULTS": str(tmp_cwd)}
    out = subprocess.run([sys.executable, str(SCRIPTS / name), *args], capture_output=True, text=True, env=env, cwd=tmp_cwd)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


@pytest.mark.slow
def test_flop_ledger_script(tmp_path):
    assert "ratio 13.8x" in run("02_flop_ledger.py", tmp_cwd=tmp_path)


@pytest.mark.network
def test_cifar_and_tiny_imagenet_loaders():
    from vitfamily.data import cifar10, tiny_imagenet_val

    x, y = cifar10("test")
    assert x.shape == (10_000, 3, 32, 32) and y.max() == 9
    x, y = tiny_imagenet_val()
    assert x.shape == (10_000, 3, 64, 64)


@pytest.mark.network
@pytest.mark.slow
def test_pretrained_deit_script(tmp_path):
    assert "top-1 agrees" in run("06_pretrained_deit.py", "--images", "10", "--batch", "10", tmp_cwd=tmp_path)
