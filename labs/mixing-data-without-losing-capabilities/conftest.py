"""Make `import mixlab` work when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The tests train models with a few thousand parameters, so two threads are plenty.
torch.set_num_threads(2)
