"""Make the C4 tutorial's `tinyllm` package importable. This lab reuses its decoder instead of rebuilding one."""

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
C04 = REPO / "c04-llm"
if str(C04) not in sys.path:
    sys.path.insert(0, str(C04))
