"""Shared setup for the scripts: make `docread` importable, and write every printed line to results/ as well."""

import json
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
RESULTS = LAB / "results"
RESULTS.mkdir(exist_ok=True)


def hardware() -> str:
    """The CPU model and thread count, never the machine's name."""
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__.split('+')[0]}"


class Report:
    """print() that also goes to results/<name>.txt, which starts with a hardware line and the command."""

    def __init__(self, name: str):
        self.path = RESULTS / f"{name}.txt"
        rel = Path(sys.argv[0]).resolve().relative_to(LAB.parents[1])
        self.lines = [f"# hardware: {hardware()}",
                      f"# command: python {rel} {' '.join(sys.argv[1:])}".rstrip(), ""]
        print("\n".join(self.lines))

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self, data: dict | None = None):
        self.path.write_text("\n".join(self.lines) + "\n")
        if data is not None:
            payload = {"hardware": hardware(), "command": self.lines[1][len("# command: "):], **data}
            self.path.with_suffix(".json").write_text(json.dumps(payload, indent=1) + "\n")
