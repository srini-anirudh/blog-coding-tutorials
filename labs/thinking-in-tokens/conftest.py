"""Make `import thinktok` (and, through it, the C4 tutorial's `tinyllm`) work when pytest runs from the repo root."""

import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
torch.set_num_threads(2)
