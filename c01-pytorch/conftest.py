"""Make `import scratchnet` work when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# These tests are tiny. Two threads are enough, and they leave the other cores free.
torch.set_num_threads(2)
