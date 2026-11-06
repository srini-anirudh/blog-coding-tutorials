"""Make `import organs` work when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import organs  # noqa: E402,F401  (also puts C3 and C4 on sys.path)

torch.set_num_threads(2)
