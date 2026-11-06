"""Make `import cnnverse` and the scripts importable when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

torch.set_num_threads(2)
