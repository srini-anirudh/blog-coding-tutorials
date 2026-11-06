"""Put this folder on sys.path so ``import agentcraft`` works when pytest runs from the repo root."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
torch.set_num_threads(2)
