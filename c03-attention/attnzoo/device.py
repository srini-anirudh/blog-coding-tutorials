"""Device and thread helpers shared by scripts, tests and benchmarks."""

from __future__ import annotations

import platform
import re
import subprocess

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device: cuda > mps > cpu, unless `prefer` names one."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued kernels, so wall-clock timings measure the work and not the launch."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def be_polite(num_threads: int = 8) -> None:
    """Cap intra-op threads, so demos and benchmarks stay light on a shared machine."""
    torch.set_num_threads(min(num_threads, torch.get_num_threads()))


def cpu_name() -> str:
    """The CPU model, read from the OS: 'AMD EPYC 7R13', 'Apple M5'. Never the machine's name."""
    if platform.system() == "Darwin":
        cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip() or "unknown CPU"
    try:
        with open("/proc/cpuinfo") as f:
            match = re.search(r"model name\s*:\s*(.+)", f.read())
    except OSError:
        match = None
    return match.group(1).replace(" Processor", "").strip() if match else platform.machine()


def hardware_note(device: str = "cpu") -> str:
    """One line naming what produced a number. Measured results must carry it."""
    if device == "cuda":
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}"
    if device == "mps":
        return f"{cpu_name()} GPU through MPS, torch {torch.__version__}"
    return f"{cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__}"
