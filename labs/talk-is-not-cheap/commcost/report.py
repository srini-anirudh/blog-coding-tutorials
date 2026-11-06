"""Write results files that start with a hardware line and the command that produced them."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parent.parent
RESULTS = LAB / "results"


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def hardware(gpu: bool = False, note: str = "") -> str:
    if gpu and torch.cuda.is_available():
        n = torch.cuda.device_count()
        hw = f"{n} x {torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
        hw += f", NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}"
    else:
        hw = f"{cpu_name()}, torch {torch.__version__}"
        try:
            hw += f", load average {os.getloadavg()[0]:.0f} on {os.cpu_count()} CPUs"
        except OSError:
            pass
    return hw + (f", {note}" if note else "")


def command() -> str:
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(LAB.parent.parent)
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


class Report:
    """Collect lines, print them as they come, and save them with the header on close."""

    def __init__(self, name: str, gpu: bool = False, note: str = ""):
        self.name = name
        self.lines = [f"# hardware: {hardware(gpu, note)}", f"# command: {command()}", ""]
        for line in self.lines[:2]:
            print(line)

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self, data: dict | list | None = None) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        if data is not None:
            payload = {"hardware": self.lines[0][len("# hardware: ") :], "command": command(), "data": data}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1, default=float) + "\n")


def table(rows: list[list], header: list[str]) -> list[str]:
    cells = [header] + [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(header))]
    fmt = lambda r: "  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r))  # noqa: E731
    return [fmt(header), fmt(["-" * x for x in w])] + [fmt(r) for r in cells[1:]]


def human_bytes(n: float) -> str:
    for unit, k in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= k:
            v = n / k
            return f"{v:.0f} {unit}" if v == int(v) else f"{v:.1f} {unit}"
    return f"{int(n)} B"


def env_threads() -> str:
    return os.environ.get("OMP_NUM_THREADS", "unset")
