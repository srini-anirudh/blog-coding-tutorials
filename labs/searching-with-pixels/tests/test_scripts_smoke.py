"""Run every script end to end at toy size (PIXSEARCH_FAST=1 writes to runs/fast/, not results/)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.slow
def test_all_scripts_run_in_fast_mode():
    env = {**os.environ, "PIXSEARCH_FAST": "1"}
    for name, extra in [("01_metrics.py", []), ("02_three_scorers.py", []), ("02b_bi_longer.py", []),
                        ("03_cascade.py", []),
                        ("04_negatives.py", []), ("05_modality_gap.py", ["--skip-clip"]), ("06_index_cost.py", [])]:
        r = subprocess.run([sys.executable, str(SCRIPTS / name), *extra], env=env, capture_output=True, text=True,
                           timeout=900)
        assert r.returncode == 0, f"{name}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"


@pytest.mark.network
@pytest.mark.slow
def test_clip_part_runs():
    env = {**os.environ, "PIXSEARCH_FAST": "1"}
    subprocess.run([sys.executable, str(SCRIPTS / "02_three_scorers.py")], env=env, check=True, capture_output=True,
                   timeout=900)
    r = subprocess.run([sys.executable, str(SCRIPTS / "05_modality_gap.py")], env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Part B." in r.stdout
