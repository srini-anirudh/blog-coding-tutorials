"""Small public image sets from the Hugging Face Hub, decoded once into uint8 tensors and cached.

cifar10        uoft-cs/cifar10, 50,000 train and 10,000 test images, 32 x 32
tiny_imagenet  zh-plus/tiny-imagenet validation split, 10,000 images, 64 x 64 (ImageNet classes)
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _cache_dir() -> Path:
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    path = root / "vitfamily"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _decode(parquet_path: str, image_col: str) -> tuple[torch.Tensor, torch.Tensor]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path).to_pydict()
    images = [np.asarray(Image.open(io.BytesIO(x["bytes"])).convert("RGB")) for x in table[image_col]]
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W) uint8
    return x, torch.tensor(table["label"])


def _load(repo: str, filename: str, image_col: str, tag: str) -> tuple[torch.Tensor, torch.Tensor]:
    cached = _cache_dir() / f"{tag}.pt"
    if cached.exists():
        return torch.load(cached)
    from huggingface_hub import hf_hub_download

    out = _decode(hf_hub_download(repo, filename, repo_type="dataset"), image_col)
    torch.save(out, cached)
    return out


def cifar10(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    return _load("uoft-cs/cifar10", f"plain_text/{split}-00000-of-00001.parquet", "img", f"cifar10_{split}")


def tiny_imagenet_val() -> tuple[torch.Tensor, torch.Tensor]:
    name = "data/valid-00000-of-00001-70d52db3c749a935.parquet"
    return _load("zh-plus/tiny-imagenet", name, "image", "tiny_imagenet_val")


CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 3, H, W) -> float, CIFAR mean/std."""
    return (x.float() / 255 - CIFAR_MEAN.to(x.device)) / CIFAR_STD.to(x.device)


def augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Float (B, C, H, W): random crop after 4-pixel reflect padding, then random horizontal flip."""
    b, c, h, w = x.shape
    dev = x.device
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    rows = torch.randint(0, 2 * pad + 1, (b, 1), device=dev) + torch.arange(h, device=dev)
    cols = torch.randint(0, 2 * pad + 1, (b, 1), device=dev) + torch.arange(w, device=dev)
    flip = torch.rand(b, 1, device=dev) < 0.5
    cols = torch.where(flip, cols.flip(-1), cols)
    bi = torch.arange(b, device=dev)[:, None, None, None]
    ci = torch.arange(c, device=dev)[None, :, None, None]
    return xp[bi, ci, rows[:, None, :, None], cols[:, None, None, :]]
