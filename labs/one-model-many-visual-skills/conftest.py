"""Make `import skillmix` work when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The tests use a model of a few hundred thousand parameters. Two threads are enough.
torch.set_num_threads(2)
