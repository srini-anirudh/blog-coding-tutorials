"""Put the tutorial folder on sys.path, so scripts run from anywhere without installing."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.set_num_threads(2)  # the laptop is shared
torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)
