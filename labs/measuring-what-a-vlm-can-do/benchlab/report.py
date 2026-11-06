"""Print a result and save the same text under results/, headed by the hardware and the command."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]


def hardware() -> str:
    """One line naming the hardware and software. No host or user names."""
    v = f"torch {torch.__version__}"
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, {v}, CUDA {torch.version.cuda}"
    cpu = platform.processor() or "CPU"
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, {v}"


def command_line() -> str:
    """The command that produced a result, with the script path relative to the repository root."""
    return "python " + " ".join([os.path.relpath(os.path.abspath(sys.argv[0]), REPO)] + sys.argv[1:])


class Report:
    def __init__(self, name: str):
        self.path = LAB / "results" / f"{name}.txt"
        self.lines = [f"# hardware: {hardware()}", f"# command: {command_line()}", ""]

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
        print(f"\nwrote {self.path.relative_to(LAB)}")
