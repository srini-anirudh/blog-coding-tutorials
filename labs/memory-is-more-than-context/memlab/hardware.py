"""A hardware line for every results file. It names the CPU or GPU model only, never the machine."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import torch


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip().replace(" Processor", "")
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def hardware_label(device: str = "cpu") -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    return f"{cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def header(device: str = "cpu") -> str:
    """Two lines: the hardware, then the command that produced the file, relative to the repo root."""
    script = Path(sys.argv[0]).name
    args = " ".join(sys.argv[1:])
    cmd = f"python labs/memory-is-more-than-context/scripts/{script} {args}".rstrip()
    return f"# hardware: {hardware_label(device)}\n# command: {cmd}\n"
