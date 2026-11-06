"""The real-model path of prior_split_smolvlm.py. Downloads SmolVLM-256M from the Hub on first run."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.network

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prior_split_smolvlm.py"


def test_prior_split_runs_on_two_scenes():
    spec = importlib.util.spec_from_file_location("prior_split_smolvlm", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    proc = mod.load_processor()
    model = mod.load_model().eval()
    res = mod.part_a(model, proc, 2, lambda *a: None)
    assert 0 <= res["acc_image"] <= 1 and set(res["vcd"]) == {str(w) for w in mod.BLENDS}
