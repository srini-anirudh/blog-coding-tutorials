"""Hardware labels and result files. Every result file starts with the hardware line and the command."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parent.parent
RESULTS = LAB / "results"


def hardware_label(device: torch.device | str = "cpu") -> str:
    """For example "AMD EPYC 7R13, 8 threads, torch 2.15.0". Never a hostname."""
    device = torch.device(device)
    version = torch.__version__.split("+")[0]
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name()}, torch {version}, CUDA {torch.version.cuda}"
    name = platform.processor() or "CPU"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip().removesuffix(" Processor")
                break
    return f"{name}, {torch.get_num_threads()} threads, torch {version}"


def command_line() -> str:
    """The command that produced a result, relative to the repository root."""
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(LAB.parent.parent)
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


class Report:
    """Collects printed lines and a JSON payload, then writes results/<name>.txt and results/<name>.json."""

    def __init__(self, name: str, device: torch.device | str = "cpu"):
        self.name = name
        self.header = [f"# hardware: {hardware_label(device)}", f"# command: {command_line()}"]
        self.lines: list[str] = []
        self.data: dict = {"hardware": hardware_label(device), "command": command_line()}

    def print(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def table(self, rows: list[dict], floatfmt: str = "{:.4f}") -> None:
        """A plain fixed-width table from a list of dicts with the same keys."""
        if not rows:
            return
        keys = list(rows[0])
        cells = [[floatfmt.format(r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys] for r in rows]
        widths = [max(len(k), *(len(c[i]) for c in cells)) for i, k in enumerate(keys)]
        self.print("  ".join(k.rjust(w) for k, w in zip(keys, widths)))
        for c in cells:
            self.print("  ".join(v.rjust(w) for v, w in zip(c, widths)))

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.header + self.lines) + "\n")
        (RESULTS / f"{self.name}.json").write_text(json.dumps(self.data, indent=1) + "\n")
