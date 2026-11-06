import importlib.util
from pathlib import Path

from whygpu import hw

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hardware_line_has_no_private_strings():
    line = hw.hardware_line("cpu")
    assert line.startswith("hardware: ") and "threads" in line
    assert "/" not in line.replace("+cu", "")


def test_report_writes_header(tmp_path, monkeypatch):
    monkeypatch.setattr(hw, "RESULTS", tmp_path)
    r = hw.Report("x.txt", "cpu")
    r.table(["a", "b"], [[1, 2.5]])
    text = r.save().read_text().splitlines()
    assert text[0].startswith("hardware:") and text[1].startswith("command: python")
    assert "2.5" in text[-1]


def test_models_script_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(hw, "RESULTS", tmp_path)
    load("00_models").main()
    out = (tmp_path / "00_models.txt").read_text()
    assert "295 FLOP per byte" in out


def test_fp16_accumulation_loses_to_fp32():
    r = load("08_tensor_cores").accumulate()
    assert r["rel_err_fp32"] < 1e-5
    assert r["rel_err_fp16"] > 100 * r["rel_err_fp32"]
