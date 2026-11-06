"""Make `import tinyvlm` work when pytest runs from the repository root."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The Hub models used by the network tests are cached after the first run; do not ping the Hub every time.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Many tutorials share one laptop; keep each test process polite.
torch.set_num_threads(2)
