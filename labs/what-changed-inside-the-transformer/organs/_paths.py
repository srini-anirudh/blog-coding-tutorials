"""Make `tinyllm` (C4) and `attnzoo` (C3) importable from this lab, wherever it is run from."""

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for folder in (REPO / "c04-llm", REPO / "c03-attention", LAB):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

RESULTS = LAB / "results"
