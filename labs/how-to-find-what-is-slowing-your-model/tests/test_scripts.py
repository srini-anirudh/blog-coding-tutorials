"""End-to-end runs of the CPU scripts. They sleep rather than compute, so they are slow but not heavy."""

import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parent.parent


def run(*args):
    p = subprocess.run([sys.executable, *args], cwd=LAB, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout


@pytest.mark.slow
def test_real_loader_matches_the_queue_model():
    out = run("scripts/04_input_pipeline.py", "--quick")
    rows = [ln.split() for ln in out.splitlines() if ln.strip()[:1].isdigit() and "ms" in ln]
    waits = {int(r[0]): float(r[1].rstrip("ms")) for r in rows}
    assert 8 < waits[4] < 30 and waits[8] < 6  # model says 16 ms and 0 ms


@pytest.mark.slow
def test_barrier_splits_skew_from_transfer():
    import json

    run("scripts/05_distributed_skew.py")
    res = json.loads((LAB / "results" / "05_distributed_skew.json").read_text())["results"]
    fast, slow = res["0"]["one rank slow"], res["2"]["one rank slow"]
    assert fast["barrier_ms"] > 12 and slow["barrier_ms"] < 6  # fast ranks wait for the slow one
