"""Small helpers shared by every stage: device choice, seeding, parameter and memory accounting."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def pick_device(prefer: str | None = None) -> torch.device:
    """Return cuda > mps > cpu, or the device named in `prefer` / $POSTTRAIN_DEVICE."""
    name = prefer or os.environ.get("POSTTRAIN_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def polite_threads(n: int = 2) -> None:
    """The laptop is shared by many runs; two CPU threads each keeps it responsive."""
    torch.set_num_threads(n)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts. Tied weights are counted once."""
    seen: set[int] = set()
    trainable = total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def device_memory_mb(device: torch.device) -> float:
    """Memory currently held by tensors on `device`, in MiB (0 on CPU, where we do not track it)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return 0.0


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def save_json(obj: object, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2))
    return path


def dir_size_mb(path: str | Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**20
