"""Put the lab folder on sys.path, and print the hardware and command, so every saved output names both."""

import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parent.parent
REPO = LAB.parent.parent
C3 = REPO / "c03-attention"  # C3 builds RoPE and 2D RoPE; this lab imports them instead of rebuilding them
sys.path.insert(0, str(LAB))

from twomod.hw import hardware_note  # noqa: E402

torch.set_printoptions(precision=4, sci_mode=False, linewidth=110)


def header(device: str = "cpu") -> None:
    script = Path(sys.argv[0]).resolve().relative_to(REPO)
    print(f"# hardware: {hardware_note(device)}")
    print(f"# command: python {script} {' '.join(sys.argv[1:])}".rstrip())


def section(title: str) -> None:
    print(f"\n== {title} ==")
