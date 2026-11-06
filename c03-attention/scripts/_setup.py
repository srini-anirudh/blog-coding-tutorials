"""Put the tutorial folder on sys.path, so scripts run from anywhere without installing."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attnzoo.device import be_polite, hardware_note  # noqa: E402

be_polite()
torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)
print(f"# {hardware_note()}")  # every saved output names the hardware that produced it
