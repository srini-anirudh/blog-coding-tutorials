"""Seeding, hardware labels and result files."""

from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

LAB_DIR = Path(__file__).resolve().parent.parent
RESULTS = LAB_DIR / "results"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hardware_line() -> str:
    """CPU model, thread count and torch version. No host names."""
    cpu = platform.processor() or "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    return f"Hardware: {cpu}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}, CPU only"


def command_line() -> str:
    script = Path(sys.argv[0])
    try:
        script = script.resolve().relative_to(LAB_DIR.parent.parent)
    except ValueError:
        script = Path(script.name)
    return "Command: python " + " ".join([str(script), *sys.argv[1:]])


class Report:
    """Print lines and keep them, then write them to results/<name>.txt and .json."""

    def __init__(self, name: str):
        self.name = name
        self.lines = [hardware_line(), command_line(), ""]
        self.data: dict = {}
        for line in self.lines[:2]:
            print(line)

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.lines.append(line)

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        payload = {"hardware": self.lines[0], "command": self.lines[1], **self.data}
        (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def ascii_image(img: torch.Tensor, chars: str = " .:-=+*#%@") -> str:
    """Render a 2D tensor with values in [0, 1] as text."""
    img = img.detach().float().clamp(0, 1)
    rows = []
    for row in img:
        rows.append("".join(chars[min(len(chars) - 1, int(v * (len(chars) - 1) + 0.5))] * 2 for v in row))
    return "\n".join(rows)
