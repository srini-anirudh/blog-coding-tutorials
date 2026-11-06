"""Run the scripts end to end with small settings. Marked slow; the network one needs the Hub (or its cache)."""

import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]


def run(script, *args, tmp):
    out = tmp / Path(script).stem
    subprocess.run([sys.executable, str(LAB / "scripts" / script), *args, "--out", str(out)], check=True,
                   capture_output=True, text=True, timeout=900)
    text = out.with_suffix(".txt").read_text()
    assert text.startswith("# hardware:")
    return text


@pytest.mark.slow
@pytest.mark.parametrize("script,args", [
    ("01_compounding_error.py", ["--steps", "200"]),
    ("02_multimodal_demos.py", ["--steps", "200"]),
    ("03_action_tokens.py", ["--n-train", "60"]),
    ("04_chunk_latency_loop.py", ["--seeds", "2"]),
    ("05_action_expert.py", ["--prefix", "64", "--layers", "2", "--reps", "1"]),
    ("06_trial_statistics.py", []),
    ("07_knowledge_insulation.py", ["--pretrain-steps", "50", "--robot-steps", "50", "--seeds", "1"]),
])
def test_script_runs(script, args, tmp_path):
    run(script, *args, tmp=tmp_path)


@pytest.mark.network
def test_vocabulary_demo(tmp_path):
    text = run("03_action_tokens.py", "--vocab", "--n-train", "60", tmp=tmp_path)
    assert "of the 256 chosen ids never occur there" in text
