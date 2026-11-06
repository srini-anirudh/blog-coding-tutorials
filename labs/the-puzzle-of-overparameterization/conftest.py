"""Make `import overparam` work when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# These tests are small. Two threads are enough on a shared machine.
torch.set_num_threads(2)
