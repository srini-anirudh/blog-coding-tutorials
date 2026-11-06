"""Make `import archlab` work when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
torch.set_num_threads(2)  # tests use tiny tensors; stay polite on a shared machine
