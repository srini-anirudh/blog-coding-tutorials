"""Name the hardware behind a number. Only the CPU or GPU model, never the machine's name."""

from __future__ import annotations

import platform
import re

import torch


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            match = re.search(r"model name\s*:\s*(.+)", f.read())
    except OSError:
        match = None
    return match.group(1).replace(" Processor", "").strip() if match else platform.machine()


def hardware_note(device: str | torch.device = "cpu") -> str:
    if torch.device(device).type == "cuda":
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    return f"{cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__}"
