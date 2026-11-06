"""Make `import agentmem` work when pytest runs from the repository root."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Many tutorials share one machine; keep each test process polite.
torch.set_num_threads(2)
