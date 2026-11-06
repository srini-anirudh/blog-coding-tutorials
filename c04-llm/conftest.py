"""Make `import tinyllm` work when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))  # for c04_helpers (uniquely named: other folders have tests/ too)

# Many tutorials share one laptop; keep each test process polite.
torch.set_num_threads(2)
