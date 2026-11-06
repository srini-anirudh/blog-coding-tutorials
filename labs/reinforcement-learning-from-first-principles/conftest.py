"""Make `import rlfp` work when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The tests use tiny models. Two threads keep them polite on a shared machine.
torch.set_num_threads(2)
