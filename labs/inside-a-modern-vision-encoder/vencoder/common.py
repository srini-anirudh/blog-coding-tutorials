"""Small helpers shared by every script: a hardware line and a results writer."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
RESULTS = LAB / "results"


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip().replace(" Processor", "")
    except OSError:
        pass
    return platform.processor() or platform.machine()


def hardware() -> str:
    """One line naming the machine. No hostname, no user."""
    parts = [f"{cpu_name()}, {torch.get_num_threads()} threads", f"torch {torch.__version__}"]
    if torch.cuda.is_available():
        parts.insert(0, f"{torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}")
    return " | ".join(parts)


def command() -> str:
    """The command that produced a result, with the script path relative to the repo root."""
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(REPO)
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


def header() -> str:
    return f"# hardware: {hardware()}\n# command: {command()}\n"


def write_text(name: str, lines: list[str]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(header() + "\n".join(lines) + "\n")
    return path


def write_json(name: str, data: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    payload = {"hardware": hardware(), "command": command(), **data}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def table(rows: list[list], header_row: list[str]) -> list[str]:
    """A plain fixed-width text table."""
    cells = [[str(c) for c in header_row]] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header_row))]
    out = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in cells]
    out.insert(1, "  ".join("-" * w for w in widths))
    return out
