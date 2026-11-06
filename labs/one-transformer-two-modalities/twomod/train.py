"""A small image-classification loop: the same data, optimizer and step budget for a ViT and a CNN.

The whole dataset lives on the device as uint8, so a step is one gather plus a random crop and flip.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CIFAR-10 from the Hugging Face Hub (uoft-cs/cifar10) -> uint8 (N, 3, 32, 32) images and int64 labels."""
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10")

    def to_tensors(split):
        images = np.stack([np.asarray(im.convert("RGB")) for im in split["img"]])
        return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(), torch.tensor(split["label"])

    return (*to_tensors(ds["train"]), *to_tensors(ds["test"]))


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, device=images.device)[:, None, None]
    std = torch.tensor(CIFAR_STD, device=images.device)[:, None, None]
    return (images.float() / 255 - mean) / std


def augment(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random crop after zero padding, and a random horizontal flip. Same for every model."""
    b, _, h, w = images.shape
    padded = F.pad(images, (pad, pad, pad, pad))
    dy = torch.randint(0, 2 * pad + 1, (b,), device=images.device)
    dx = torch.randint(0, 2 * pad + 1, (b,), device=images.device)
    dev = images.device
    rows = (dy[:, None] + torch.arange(h, device=dev))[:, None, :, None]  # (B, 1, H, 1)
    cols = (dx[:, None] + torch.arange(w, device=dev))[:, None, None, :]  # (B, 1, 1, W)
    out = padded[torch.arange(b, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None],
                 rows, cols]
    flip = torch.rand(b, device=images.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


@dataclass
class TrainConfig:
    steps: int = 4000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    seed: int = 0


def lr_at(step: int, cfg: TrainConfig) -> float:
    warm = max(1, int(cfg.warmup_frac * cfg.steps))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, cfg.steps - warm)))


@torch.no_grad()
def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, batch_size: int = 1000) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(images), batch_size):
        logits = model(normalize(images[i : i + batch_size]))
        correct += (logits.argmax(-1) == labels[i : i + batch_size]).sum().item()
    model.train()
    return correct / len(images)


def fit(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    cfg: TrainConfig,
) -> dict:
    """Train for a fixed number of steps (the same compute whatever the dataset size). Returns a summary."""
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr
    )
    model.train()
    t0, losses = time.time(), []
    for step in range(cfg.steps):
        idx = torch.randint(0, len(train_x), (cfg.batch_size,), generator=gen).to(train_x.device)
        x, y = normalize(augment(train_x[idx])), train_y[idx]
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == cfg.steps - 1:
            losses.append(loss.item())
    seconds = time.time() - t0
    return {
        "train_images": len(train_x),
        "steps": cfg.steps,
        "final_train_loss": round(losses[-1], 4),
        "train_acc": round(evaluate(model, train_x[:10_000], train_y[:10_000]), 4),
        "test_acc": round(evaluate(model, test_x, test_y), 4),
        "seconds": round(seconds, 1),
    }
