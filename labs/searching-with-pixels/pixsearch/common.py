"""Paths, a hardware line for every result file, and saving and loading small checkpoints."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
# PIXSEARCH_FAST=1 (the smoke test) writes to runs/fast/ so it never overwrites the real outputs.
_FAST = os.environ.get("PIXSEARCH_FAST") == "1"
RUNS = LAB / "runs" / "fast" if _FAST else LAB / "runs"           # checkpoints, not in git
RESULTS = RUNS / "results" if _FAST else LAB / "results"


def hardware_line() -> str:
    cpu = platform.processor() or "unknown CPU"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                    break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, CPU only, torch {torch.__version__}, float32"


def command_line() -> str:
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(LAB.parents[1])
    except ValueError:
        script = Path(script.name)
    return " ".join(["python", str(script), *sys.argv[1:]])


class Report:
    """Collects printed lines and writes them to results/<name>.txt with a header."""

    def __init__(self, name: str):
        self.name = name
        self.lines = [f"# hardware: {hardware_line()}", f"# command: {command_line()}", ""]

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self, data: dict | None = None) -> None:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")
        if data is not None:
            payload = {"hardware": hardware_line(), "command": command_line(), **data}
            (RESULTS / f"{self.name}.json").write_text(json.dumps(payload, indent=1) + "\n")


def save_model(model: torch.nn.Module, name: str, meta: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "meta": meta}, RUNS / f"{name}.pt")


def load_model(name: str, build) -> tuple[torch.nn.Module, dict]:
    ck = torch.load(RUNS / f"{name}.pt", map_location="cpu", weights_only=True)
    model = build(ck["meta"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["meta"]


def fast_mode() -> bool:
    """PIXSEARCH_FAST=1 shrinks every experiment (used by the slow smoke test)."""
    return os.environ.get("PIXSEARCH_FAST") == "1"
