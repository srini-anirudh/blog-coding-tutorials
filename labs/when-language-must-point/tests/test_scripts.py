"""The four fast scripts run end to end (no training). They write to a temporary results folder."""

import runpy
import sys
from pathlib import Path

import pytest

import pointlab.report as report

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize("name", ["01_shapes.py", "03_quantisation.py", "04_ce_blindness.py", "07_metrics.py"])
def test_script_runs(name, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPTS / name)])
    runpy.run_path(str(SCRIPTS / name), run_name="__main__")
    assert any(tmp_path.iterdir())


@pytest.mark.network
def test_encodings_script_with_real_tokenizers(tmp_path, monkeypatch):
    """Needs the SmolLM2 and Qwen2.5 tokenizers (downloaded from the Hugging Face Hub on first use)."""
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPTS / "02_encodings.py")])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    runpy.run_path(str(SCRIPTS / "02_encodings.py"), run_name="__main__")
    assert "tokens_SmolLM2-135M" in (tmp_path / "02_encodings.json").read_text()
