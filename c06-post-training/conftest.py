"""Make `import posttrain` work when pytest runs from the repository root."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The tests use small models, so two threads are enough. It keeps them polite on a shared machine.
torch.set_num_threads(2)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
