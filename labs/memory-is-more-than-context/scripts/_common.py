"""Shared helpers for the scripts: import path, results folder, and a tee that writes the results file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
RESULTS = LAB / "results"

from memlab.hardware import header  # noqa: E402


class Report:
    """Collect printed lines and save them to results/<name>.txt with the hardware and command on top."""

    def __init__(self, name: str, device: str = "cpu"):
        self.name, self.device, self.lines = name, device, []

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self, data: dict | None = None) -> None:
        RESULTS.mkdir(exist_ok=True)
        head = header(self.device)
        (RESULTS / f"{self.name}.txt").write_text(head + "\n".join(self.lines) + "\n")
        if data is not None:
            data = {"hardware": head.splitlines()[0][len("# hardware: "):],
                    "command": head.splitlines()[1][len("# command: "):], **data}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(data, indent=1) + "\n")
