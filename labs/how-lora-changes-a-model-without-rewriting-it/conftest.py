"""Make `import loralab` work when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The tests are small. Two threads are enough and leave the other cores free.
torch.set_num_threads(2)
