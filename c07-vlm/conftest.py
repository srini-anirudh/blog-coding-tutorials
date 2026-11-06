"""Make `import tinyvlm` work when pytest runs from the repository root."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# The offline tests use tiny models, which gain nothing from more threads.
torch.set_num_threads(2)
