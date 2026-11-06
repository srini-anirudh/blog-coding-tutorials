"""Write each script's output to results/, starting with a hardware line and the command."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"


def hardware_label(device: str = "cpu") -> str:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    threads = os.environ.get("OMP_NUM_THREADS", str(torch.get_num_threads()))
    cpu = "AMD EPYC 7R13" if "EPYC" in _cpu_model() else (_cpu_model() or platform.processor() or "CPU")
    return f"{cpu}, {threads} threads, torch {torch.__version__}"


def _cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def command() -> str:
    """The command as a reader would type it from the repository root."""
    script = Path(sys.argv[0]).resolve()
    try:
        rel = script.relative_to(LAB.parents[1])
    except ValueError:
        rel = Path(script.name)
    return " ".join(["python", str(rel)] + sys.argv[1:])


class Report:
    """print() to stdout and to results/<name>.txt."""

    def __init__(self, name: str, device: str = "cpu"):
        RESULTS.mkdir(exist_ok=True)
        self.name = name
        self.path = RESULTS / f"{name}.txt"
        self.hardware = hardware_label(device)
        self.lines = [f"# hardware: {self.hardware}", f"# command: {command()}"]
        for line in self.lines:
            print(line)

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)
        self.path.write_text("\n".join(self.lines) + "\n")

    def json(self, data: dict) -> None:
        out = {"hardware": self.hardware, "command": command(), **data}
        (RESULTS / f"{self.name}.json").write_text(json.dumps(out, indent=1) + "\n")
