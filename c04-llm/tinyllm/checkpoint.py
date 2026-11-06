"""Native checkpoint: a folder with config.json and model.safetensors."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .config import ModelConfig
from .model import TinyLLM


def save_checkpoint(model: TinyLLM, path: str | Path) -> Path:
    """Write config.json and model.safetensors into the folder `path`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.config.save(path / "config.json")
    state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    if model.config.tie_word_embeddings:
        # Same storage as embed_tokens.weight. safetensors refuses shared tensors, and the
        # duplicate would only waste V*D values on disk, so drop it and re-tie on load.
        del state["lm_head.weight"]
    save_file(state, path / "model.safetensors")
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> TinyLLM:
    """Rebuild the model from config.json and load the weights. Fails loudly on any mismatch."""
    path = Path(path)
    config = ModelConfig.load(path / "config.json")
    model = TinyLLM(config)
    state = load_file(path / "model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device)
