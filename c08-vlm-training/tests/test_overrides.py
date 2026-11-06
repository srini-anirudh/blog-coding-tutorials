import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train.py"


def _load():
    spec = importlib.util.spec_from_file_location("train_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scientific_notation_overrides_are_floats_not_strings():
    out = _load().parse_overrides(["lrs.vision=1e-5", "lrs.lm=5e-5", "steps=200", "name=run1"])
    assert out == {"lrs": {"vision": 1e-5, "lm": 5e-5}, "steps": 200, "name": "run1"}
    assert isinstance(out["lrs"]["vision"], float)


def test_lists_dicts_and_bools_still_parse():
    out = _load().parse_overrides(["tasks=[doc,ocr]", "manual_weights={caption: 1, vqa: 2}", "save=true"])
    assert out == {"tasks": ["doc", "ocr"], "manual_weights": {"caption": 1, "vqa": 2}, "save": True}
