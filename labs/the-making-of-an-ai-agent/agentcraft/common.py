"""Small helpers shared by every script: a hardware line, and writing results files that start with it."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
RESULTS = LAB / "results"


def hardware(device: str = "cpu") -> str:
    """'AMD EPYC 7R13 Processor, 8 threads, torch 2.15.0, CPU only', or the GPU name with torch and CUDA versions.
    No hostname, no user."""
    import torch

    if device.startswith("cuda"):
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__.split('+')[0]}, CUDA {torch.version.cuda}"

    cpu = platform.processor() or "unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    cpu = cpu.replace(" 48-Core Processor", "").replace(" 64-Core Processor", "").replace(" 96-Core Processor", "")
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}, CPU only"


def command() -> str:
    """The command that is running, relative to the repository root."""
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(REPO)
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


def write_results(name: str, text: str, data: dict | None = None, device: str = "cpu") -> None:
    """results/<name>.txt (and .json): the hardware line and the command come first."""
    RESULTS.mkdir(exist_ok=True)
    head = f"# hardware: {hardware(device)}\n# command: {command()}\n\n"
    (RESULTS / f"{name}.txt").write_text(head + text.rstrip() + "\n")
    if data is not None:
        payload = {"hardware": hardware(device), "command": command(), **data}
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")


def table(rows: list[list], header: list[str]) -> str:
    """A plain Markdown table."""
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)
