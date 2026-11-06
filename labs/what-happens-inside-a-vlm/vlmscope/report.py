"""Results files: every file starts with the hardware line and the command that wrote it."""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch
import transformers

RESULTS = Path(os.environ.get("VLMSCOPE_RESULTS", Path(__file__).resolve().parent.parent / "results"))


def hardware() -> str:
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        return f"{dev}, torch {torch.__version__}, CUDA {torch.version.cuda}, transformers {transformers.__version__}"
    cpu = platform.processor() or "CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().removesuffix(" Processor")
                break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__}, transformers {transformers.__version__}"


def command() -> str:
    script = Path(sys.argv[0])
    rel = f"labs/{script.resolve().parent.parent.name}/scripts/{script.name}"
    return " ".join(["python", rel, *sys.argv[1:]])


class Report:
    """Collects printed lines and a JSON payload, then writes results/<name>.txt and results/<name>.json."""

    def __init__(self, name: str):
        self.name = name
        self.lines = [f"# hardware: {hardware()}", f"# command: {command()}"]
        self.data: dict = {"hardware": hardware(), "command": command()}
        for line in self.lines:
            print(line)

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        (RESULTS / f"{self.name}.json").write_text(json.dumps(self.data, indent=1) + "\n")
        print(f"wrote results/{self.name}.txt and .json")


def table(rows: list[list], header: list[str]) -> list[str]:
    cells = [[str(c) for c in r] for r in [header, *rows]]
    w = [max(len(r[i]) for r in cells) for i in range(len(header))]
    fmt = lambda r: "  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r))
    return [fmt(cells[0]), "  ".join("-" * x for x in w), *[fmt(r) for r in cells[1:]]]
