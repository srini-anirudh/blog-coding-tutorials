"""Full training checkpoints: atomic writes, checksums, and fallback on corruption.

A checkpoint that can only restore the weights restarts training, it does not resume it.
To resume we need all of this:

    model       weights
    optimizer   AdamW's first and second moments (2x the model size) and step counts
    scheduler   where we are on the LR curve
    step        optimiser steps taken
    rng         every random generator
    data        loader position: seed, epoch, batches consumed
    config      the run definition, so a resume cannot silently change it
    extra       anything stateful in the loop, e.g. the loss-spike detector
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from pretrain.rng import rng_state, set_rng_state


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(out_dir: str | Path, step: int, model, opt, sched, loader, config: dict,
                    extra: dict | None = None, keep: int = 2) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": step,
        "rng": rng_state(),
        "data": loader.state_dict(),
        "config": config,
        "extra": extra or {},
    }
    path = out_dir / f"ckpt_{step:06d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())  # the bytes are on disk before we publish the name
    # rename is atomic: readers see the old file set or the new one, never half a file.
    os.replace(tmp, path)
    manifest = {"file": path.name, "step": step, "sha256": sha256(path), "bytes": path.stat().st_size}
    path.with_suffix(".json").write_text(json.dumps(manifest))
    for old in sorted(out_dir.glob("ckpt_*.pt"))[:-keep]:
        old.unlink()
        old.with_suffix(".json").unlink(missing_ok=True)
    return path


def verify(path: Path) -> bool:
    """True when the manifest exists and the file's checksum matches it."""
    manifest = path.with_suffix(".json")
    if not manifest.exists():
        return False
    return json.loads(manifest.read_text())["sha256"] == sha256(path)


def latest_valid_checkpoint(out_dir: str | Path) -> Path | None:
    """Newest checkpoint that passes its checksum. Corrupt ones are reported and skipped."""
    for path in sorted(Path(out_dir).glob("ckpt_*.pt"), reverse=True):
        if verify(path):
            return path
        print(f"[checkpoint] {path.name} failed verification, trying an older one")
    return None


def load_checkpoint(path: Path, model, opt, sched, loader, device: torch.device) -> dict:
    """Restore everything in place. Returns the raw state (step, config, extra)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])  # moves moment tensors to the params' device
    sched.load_state_dict(state["scheduler"])
    loader.load_state_dict(state["data"])
    set_rng_state(state["rng"])
    return state
