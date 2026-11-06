"""Shared helpers for the scripts: import path, results folder, and a writer that adds the header."""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from vlmscale.hw import header  # noqa: E402

RESULTS = LAB / "results"


class Report:
    """Print as we go, and save the same text to results/<name>.txt with the hardware line on top."""

    def __init__(self, name: str, device: str = "cpu"):
        self.path = RESULTS / f"{name}.txt"
        self.lines = [header(device)]
        print(self.lines[0])

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
        print(f"[saved {self.path.relative_to(LAB.parent.parent)}]")


def save_json(name: str, payload: dict, device: str = "cpu") -> None:
    RESULTS.mkdir(exist_ok=True)
    hw, cmd = header(device).strip().split("\n")
    out = {"hardware": hw.removeprefix("Hardware: "), "command": cmd.removeprefix("Command: "), **payload}
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")


def table(rows: list[list], headers: list[str]) -> str:
    cells = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    fmt = lambda r: "  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths)))  # noqa: E731
    return "\n".join([fmt(cells[0]), "  ".join("-" * w for w in widths)] + [fmt(r) for r in cells[1:]])
