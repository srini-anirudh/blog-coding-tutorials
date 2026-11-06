"""Small helpers shared by every stage: device choice, seeding, parameter and memory accounting."""

from __future__ import annotations

import json
import os
import platform
import random
import resource
import sys
from pathlib import Path

import torch

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
# Results land in results/, or in a subfolder named by $POSTTRAIN_RESULTS (results/h100 for the GPU runs), so runs on
# different hardware never overwrite each other.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / os.environ.get("POSTTRAIN_RESULTS", "")


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


def polite_threads(n: int | None = None) -> None:
    """Cap CPU threads so one run does not take over a shared machine: $OMP_NUM_THREADS if set, otherwise 8."""
    torch.set_num_threads(n or int(os.environ.get("OMP_NUM_THREADS", 8)))
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
    """Memory in MiB: peak allocated on CUDA, currently allocated on MPS, peak resident set of this process on CPU.

    The CPU number includes Python and the libraries, so it only compares runs made in separate processes.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux, bytes on macOS
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def hardware_label(device: torch.device) -> str:
    """The name that goes next to every number, for example "AMD EPYC 7R13, 8 threads" or "NVIDIA H100"."""
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name()}, torch {torch.__version__.split('+')[0]}, CUDA {torch.version.cuda}"
    if device.type == "mps":
        return "Apple MPS"
    name = platform.processor() or "CPU"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip().removesuffix(" Processor")
                break
    return f"{name}, {torch.get_num_threads()} threads"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def save_json(obj: object, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2))
    return path


def dir_size_mb(path: str | Path, pattern: str = "*.safetensors") -> float:
    """Size of the weight files under `path`, in MiB. Tokenizer and config files are not counted."""
    return sum(f.stat().st_size for f in Path(path).rglob(pattern) if f.is_file()) / 2**20
