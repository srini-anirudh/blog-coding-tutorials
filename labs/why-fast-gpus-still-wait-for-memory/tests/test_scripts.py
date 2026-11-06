"""The CPU scripts run and print the post's headline numbers."""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def run(name, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args], capture_output=True, text=True,
                          check=True, timeout=120).stdout


def test_ridge_points_script(tmp_path):
    out = run("01_ridge_points.py")
    for s in ("H100 SXM5", "295", "41.8 ms", "B* = s I* / 2 = 295", "T = 590"):
        assert s in out


def test_tile_traffic_script():
    out = run("02_tile_traffic.py", "--n", "1024", "--wave", "16")
    assert "x compulsory" in out
