"""Real images and real encoders: a small slice of mini-ImageNet and DINOv2 from the Hugging Face Hub.

The images are read straight from the dataset's parquet files over HTTP, a few row groups at a time, and cached
locally in `data_cache/` (not in git). Nothing here runs in the offline tests.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .preprocess import NORMS

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
DATASET = "datasets/timm/mini-imagenet/data"


def load_images(row_groups: dict[str, list[int]], per_group: int = 100) -> list[tuple[Image.Image, int]]:
    """{parquet file: [row group ids]} -> [(RGB image, label)]. Cached as raw JPEG bytes."""
    CACHE.mkdir(exist_ok=True)
    key = CACHE / ("mini-imagenet-" + "-".join(f"{f.split('-')[1]}{g}" for f, gs in row_groups.items() for g in gs)[:120] + ".npz")
    if not key.exists():
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        blobs, labels = [], []
        for fname, groups in row_groups.items():
            with fs.open(f"{DATASET}/{fname}", "rb") as f:
                pf = pq.ParquetFile(f)
                for g in groups:
                    t = pf.read_row_group(g).to_pylist()[:per_group]
                    blobs += [r["image"]["bytes"] for r in t]
                    labels += [r["label"] for r in t]
        np.savez(key, blobs=np.array(blobs, dtype=object), labels=np.array(labels))
    d = np.load(key, allow_pickle=True)
    return [(Image.open(io.BytesIO(b)).convert("RGB"), int(lab)) for b, lab in zip(d["blobs"], d["labels"])]


def dinov2_input(img: Image.Image, norm: str = "ImageNet", resize: str = "pil_bicubic", crop: bool = True,
                 size: int = 224) -> torch.Tensor:
    """DINOv2's eval transform (short side to 256, centre crop 224, ImageNet constants), with one knob broken
    at a time. resize: pil_bicubic | pil_nearest | torch_no_aa. crop=False squashes the whole image to 224."""
    w, h = img.size
    if crop:
        s = 256 / min(w, h)
        target = (max(size, round(w * s)), max(size, round(h * s)))
    else:
        target = (size, size)
    if resize == "torch_no_aa":
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)[None]
        t = F.interpolate(t, size=(target[1], target[0]), mode="bicubic", align_corners=False, antialias=False)[0]
        t = t.clamp(0, 1)
    else:
        method = Image.Resampling.BICUBIC if resize == "pil_bicubic" else Image.Resampling.NEAREST
        t = torch.from_numpy(np.asarray(img.resize(target, method), dtype=np.float32) / 255).permute(2, 0, 1)
    if crop:
        top, left = (t.shape[1] - size) // 2, (t.shape[2] - size) // 2
        t = t[:, top: top + size, left: left + size]
    mean, std = NORMS[norm]
    return (t - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


def load_dinov2(name: str):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(name).eval()
    return model, int(getattr(model.config, "num_register_tokens", 0) or 0)


@torch.no_grad()
def encode(model, pixels: torch.Tensor, batch: int = 32) -> dict[str, torch.Tensor]:
    """CLS (after the final norm) and the last block's patch tokens before the final norm."""
    cls, last = [], []
    for i in range(0, len(pixels), batch):
        o = model(pixel_values=pixels[i: i + batch], output_hidden_states=True)
        cls.append(o.pooler_output)
        last.append(o.hidden_states[-1])
    return {"cls": torch.cat(cls), "last_hidden": torch.cat(last)}


@torch.no_grad()
def token_norms(model, pixels: torch.Tensor, registers: int, batch: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch-token norms after every block, before the final norm: layers x images x patches.
    Also returns the register tokens' norms (layers x images x registers)."""
    pn, rn = [], []
    for i in range(0, len(pixels), batch):
        hs = model(pixel_values=pixels[i: i + batch], output_hidden_states=True).hidden_states[1:]
        pn.append(torch.stack([h[:, 1 + registers:].norm(dim=-1) for h in hs]))
        rn.append(torch.stack([h[:, 1: 1 + registers].norm(dim=-1) for h in hs]))
    return torch.cat(pn, dim=1), torch.cat(rn, dim=1)


def knn_accuracy(features: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    """Leave-one-out k-nearest-neighbour top-1 with cosine similarity and majority vote."""
    f = F.normalize(features, dim=-1)
    sim = f @ f.T
    sim.fill_diagonal_(-2)
    idx = sim.topk(k, dim=1).indices
    votes = labels[idx]
    pred = torch.mode(votes, dim=1).values
    return float((pred == labels).float().mean())


def outlier_stats(patch_norms: torch.Tensor, factor: float) -> dict[str, float]:
    """patch_norms: images x patches. A token is an outlier when its norm exceeds factor x the image's median."""
    med = patch_norms.median(dim=1, keepdim=True).values
    out = patch_norms > factor * med
    ratio = (patch_norms.max(dim=1).values / med[:, 0])
    return {
        "outlier_fraction": float(out.float().mean()),
        "images_with_outliers": float(out.any(dim=1).float().mean()),
        "max_over_median_mean": float(ratio.mean()),
        "max_over_median_max": float(ratio.max()),
    }


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
