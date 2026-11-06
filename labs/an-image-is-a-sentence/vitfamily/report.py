"""Write a script's output to results/: hardware line first, then the command, then the lines."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .c03 import hardware_note

# Tests point VITFAMILY_RESULTS at a temporary folder so they never overwrite the real results.
RESULTS = Path(os.environ.get("VITFAMILY_RESULTS", Path(__file__).resolve().parents[1] / "results"))


class Report:
    def __init__(self, name: str, device: str = "cpu") -> None:
        self.name = name
        self.lines: list[str] = []
        self.data: dict = {}
        args = " ".join(sys.argv[1:])
        script = f"labs/an-image-is-a-sentence/scripts/{Path(sys.argv[0]).name}"
        self.header = [f"hardware: {hardware_note(device)}", f"command: python {script} {args}".rstrip()]
        for line in self.header:
            print(line)

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.lines.append(line)

    def save(self) -> Path:
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"{self.name}.txt"
        path.write_text("\n".join(self.header + [""] + self.lines) + "\n")
        if self.data:
            payload = {"hardware": self.header[0][10:], "command": self.header[1][9:], **self.data}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1) + "\n")
        return path
