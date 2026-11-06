"""Make `import hfstack` work when pytest runs from the repository root."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The test models are tiny, so two threads are enough, and they leave the machine free for other work.
torch.set_num_threads(2)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
