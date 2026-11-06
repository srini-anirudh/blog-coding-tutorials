"""Two CPU processes under DDP against one process with the same global batch."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_two_processes_equal_one():
    env = dict(os.environ, OMP_NUM_THREADS="1")
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_port", "29402",
        str(ROOT / "scripts" / "09_accelerate.py"), "--ddp",
    ]
    out = subprocess.run(command, capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    match = re.search(r"max \|loss diff\| ([0-9.e+-]+), max \|grad norm diff\| ([0-9.e+-]+)", out.stdout)
    assert match, out.stdout
    assert float(match.group(1)) < 1e-4 and float(match.group(2)) < 1e-4
