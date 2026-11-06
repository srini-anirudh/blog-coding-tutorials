"""Make `import pixsearch` work when pytest runs from the repository root."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Toy models gain nothing from more threads, and extra threads crawl on a busy machine.
torch.set_num_threads(1)
