"""Reuse the reward primitives that C9 already builds and tests.

C9 (`c09-vlm-post-training/vlmalign/rewards.py`) writes exact match, box IoU,
mask IoU and Dice, CER and ANLS. This lab does not rebuild them. The file is
loaded by path, so the lab does not need C9's package on `sys.path`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[3] / "c09-vlm-post-training" / "vlmalign" / "rewards.py"


def _load():
    spec = importlib.util.spec_from_file_location("c09_rewards", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rewards = _load()
