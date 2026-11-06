"""Device selection shared by every script: CUDA if present, then Apple MPS, then CPU."""

import platform
from pathlib import Path

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or `prefer` if it is given ("cpu", "mps", "cuda")."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cpu_name() -> str:
    """The CPU model: /proc/cpuinfo on Linux, platform.processor() elsewhere."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].replace(" Processor", "").strip()
    return platform.processor() or "CPU"


def describe_device(device: torch.device) -> str:
    """A label for result tables, e.g. "AMD EPYC 7R13, 8 threads", "MPS", or a GPU with versions."""
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(device)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    if device.type == "cpu":
        return f"{cpu_name()}, {torch.get_num_threads()} threads"
    return device.type.upper()
