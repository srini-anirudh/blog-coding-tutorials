"""One line naming the hardware, for the top of every results file. No host names."""

from __future__ import annotations

import platform
from pathlib import Path
import sys

import torch


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip().removesuffix(" Processor")
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def hardware_line(device: str = "cpu") -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return (f"Hardware: {torch.cuda.get_device_name(0)}, torch {torch.__version__.split('+')[0]}, "
                f"CUDA {torch.version.cuda}")
    return f"Hardware: {cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}"


def header(device: str = "cpu") -> str:
    """Hardware line plus the command that produced the file, with the path made relative."""
    argv = list(sys.argv)
    if argv and argv[0]:
        parts = Path(argv[0]).resolve().parts
        if "labs" in parts:  # print the path from the repository root, never an absolute path
            argv[0] = "/".join(parts[parts.index("labs"):])
    return f"{hardware_line(device)}\nCommand: python {' '.join(argv)}\n"
