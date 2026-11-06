from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))


def load_script(name: str):
    """Import scripts/<name>.py as a module, so tests can call its functions."""
    path = LAB / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lab_nonlin_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script():
    return load_script
