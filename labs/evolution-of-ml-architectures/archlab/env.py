"""Shared script setup: polite threads, fixed seeds, and a results file that names its hardware."""

from __future__ import annotations

import platform
import re
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parent.parent
RESULTS = LAB / "results"


def cpu_name() -> str:
    """The CPU model from the OS, for example 'AMD EPYC 7R13'. Never the machine's name."""
    try:
        text = Path("/proc/cpuinfo").read_text()
        match = re.search(r"model name\s*:\s*(.+)", text)
        if match:
            return match.group(1).replace(" Processor", "").strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def hardware_note(device: str = "cpu") -> str:
    if device == "cuda":
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    return f"{cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__}"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def start(name: str, threads: int = 8, seed: int = 0, device: str = "cpu") -> None:
    """Call once at the top of a script. Output goes to stdout and to results/<name>.txt."""
    torch.set_num_threads(min(threads, torch.get_num_threads()))
    torch.manual_seed(seed)
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=110)
    RESULTS.mkdir(exist_ok=True)
    out = open(RESULTS / f"{name}.txt", "w")
    sys.stdout = _Tee(sys.__stdout__, out)
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(LAB.parent.parent)  # path from the repository root
    except ValueError:
        script = Path(script.name)
    print(f"# {hardware_note(device)}")
    print(f"# command: python {' '.join([str(script), *sys.argv[1:]])}")
    print()
