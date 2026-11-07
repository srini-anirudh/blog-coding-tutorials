"""Make `import prefopt` work when pytest runs from the repository root."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

torch.set_num_threads(2)  # the tests are small; stay polite on a shared machine
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
