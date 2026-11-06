"""What `save_pretrained` writes, and how safetensors files are laid out on disk.

A safetensors file is:
    [8 bytes: little-endian u64 header length H] [H bytes: JSON header] [raw tensor bytes]
The header maps each tensor name to its dtype, shape and byte range. Nothing is pickled,
so loading never runs code, and a loader can memory-map just the tensors it needs.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open


def list_files(directory: str | Path) -> list[tuple[str, int]]:
    """(file name, size in bytes) for every file `save_pretrained` wrote."""
    directory = Path(directory)
    return sorted((p.name, p.stat().st_size) for p in directory.iterdir() if p.is_file())


def read_safetensors_header(path: str | Path) -> dict:
    """Parse the JSON header by hand, without the safetensors library."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return header


def read_tensor_by_hand(path: str | Path, name: str) -> torch.Tensor:
    """Read one tensor using only the header offsets: seek, read bytes, reinterpret."""
    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64}
    header = read_safetensors_header(path)
    entry = header[name]
    start, stop = entry["data_offsets"]  # relative to the end of the header
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        f.seek(8 + header_len + start)
        raw = bytearray(f.read(stop - start))
    return torch.frombuffer(raw, dtype=dtypes[entry["dtype"]]).reshape(entry["shape"])


def read_index(directory: str | Path) -> dict | None:
    """The shard index (`model.safetensors.index.json`) maps every tensor name to the shard holding it."""
    path = Path(directory) / "model.safetensors.index.json"
    return json.loads(path.read_text()) if path.exists() else None


def tensor_names(path: str | Path) -> list[str]:
    with safe_open(str(path), framework="pt") as f:
        return list(f.keys())
