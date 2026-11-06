"""Make `import groundpref` work when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Tests use tiny models, so two threads are enough and keep a shared machine free.
torch.set_num_threads(2)
