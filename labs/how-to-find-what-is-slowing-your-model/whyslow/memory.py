"""Memory diagnosis: allocated, reserved, peak, and reading the OOM message properly."""

from __future__ import annotations

import re

import torch

_UNITS = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40}
_SIZE = r"([\d.]+) (B|KiB|MiB|GiB|TiB)"
_FIELDS = {
    "requested": rf"Tried to allocate {_SIZE}",
    "capacity": rf"total capacity of {_SIZE}",
    "free": rf"of which {_SIZE} is free",
    "allocated": rf"{_SIZE} is allocated by PyTorch",
    "reserved_unallocated": rf"{_SIZE} is reserved by PyTorch but unallocated",
}


def parse_oom(msg: str) -> dict[str, int]:
    """Pull the four numbers out of a PyTorch CUDA OOM message, in bytes. Missing fields are left out."""
    out = {}
    for key, pat in _FIELDS.items():
        m = re.search(pat, msg)
        if m:
            out[key] = int(float(m.group(1)) * _UNITS[m.group(2)])
    return out


def diagnose_oom(f: dict[str, int], limit: int | None = None) -> str:
    """Capacity or fragmentation. `limit` is the cap the allocator works under (a memory fraction), if any:
    under a cap the device can have plenty free while this process has none left to take."""
    room = (limit - f["allocated"] - f["reserved_unallocated"]) if limit else f.get("free", 0)
    if f.get("reserved_unallocated", 0) >= f["requested"] and room < f["requested"]:
        return "fragmentation: enough cached-but-idle memory, in pieces too small; fix the allocator or shapes"
    return "capacity: the tensors really do not fit; shard, checkpoint or shrink"


def stats(device=None) -> dict[str, int]:
    """The three numbers the post names, plus the cached-but-idle gap between reserved and allocated."""
    a = torch.cuda.memory_allocated(device)
    r = torch.cuda.memory_reserved(device)
    return {"allocated": a, "reserved": r, "peak": torch.cuda.max_memory_allocated(device), "cached_idle": r - a}


def gib(n: int) -> str:
    return f"{n / (1 << 30):.2f} GiB"
