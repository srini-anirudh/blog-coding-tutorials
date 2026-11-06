"""Put the lab folder on sys.path so `import rlvrlab` works from any directory."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.set_num_threads(min(4, torch.get_num_threads()))
