"""Shared helpers: import C7's `tinyvlm`, label results with their hardware, write result files."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
RESULTS = LAB / "results"
RUNS = LAB / "runs"  # checkpoints live here; not in git

# C7 builds the VLM we reuse: SigLIP-base + MLP projector + SmolLM2-135M-Instruct.
C07 = REPO / "c07-vlm"
if str(C07) not in sys.path:
    sys.path.insert(0, str(C07))


def hardware(device: torch.device | str = "cpu") -> str:
    """One line naming the hardware, never the machine."""
    device = torch.device(device)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"{name}, torch {torch.__version__}, CUDA {torch.version.cuda}"
    cpu = "CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        cpu = platform.processor() or "CPU"
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def command_line() -> str:
    """The command that produced a result, with the path relative to the repository root."""
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(REPO)
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


class Report:
    """Collects printed lines; `save` writes them to results/<name>.txt with a hardware header."""

    def __init__(self, name: str, device: torch.device | str = "cpu"):
        self.name, self.lines = name, [f"# hardware: {hardware(device)}", f"# command: {command_line()}"]
        print(self.lines[0])

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self, data: dict | None = None) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        if data is not None:
            payload = {"hardware": self.lines[0][len("# hardware: "):], "command": self.lines[1][len("# command: "):]}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(payload | data, indent=1) + "\n")


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
