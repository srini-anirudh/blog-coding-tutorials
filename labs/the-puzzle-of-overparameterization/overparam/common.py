"""Small helpers shared by every script: a hardware line and a results writer."""

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

RESULTS = Path(__file__).resolve().parents[1] / "results"


def hardware_line() -> str:
    """One line naming the CPU, the thread count and the library versions."""
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                    break
    except OSError:
        pass
    threads = torch.get_num_threads()
    return f"hardware: {cpu}, {threads} threads, torch {torch.__version__}, numpy {np.__version__}"


def command_line() -> str:
    """The command that produced a result file, with the lab-relative script path."""
    script = Path(sys.argv[0])
    try:
        script = script.resolve().relative_to(Path(__file__).resolve().parents[3])
    except ValueError:
        script = Path(script.name)
    return "command: python " + " ".join([str(script)] + sys.argv[1:])


def table(rows, cols):
    """Format a list of dicts as aligned text. `cols` maps a key to a format spec, e.g. {"p": "d"}."""
    def cell(v, spec):
        if v is None:
            return "-"
        return format(v, spec) if spec else str(v)
    body = [[cell(r.get(k), spec) for k, spec in cols.items()] for r in rows]
    widths = [max(len(k), *(len(b[i]) for b in body)) for i, k in enumerate(cols)]
    lines = ["  ".join(k.rjust(w) for k, w in zip(cols, widths))]
    lines += ["  ".join(c.rjust(w) for c, w in zip(b, widths)) for b in body]
    return "\n".join(lines)


class Report:
    """Print lines as they are produced, and save them to results/<name>.txt and .json at the end."""

    def __init__(self, name: str):
        self.name = name
        self.lines = [hardware_line(), command_line(), ""]
        self.data: dict = {}
        for line in self.lines[:2]:
            print(line)

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.lines.append(line)

    def save(self) -> None:
        if os.environ.get("OVERPARAM_NO_SAVE"):
            return
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        if self.data:
            payload = {"hardware": self.lines[0], "command": self.lines[1], **self.data}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1) + "\n")
