"""The full training loop, with nothing hidden.

One step is five lines: zero_grad, forward, loss, backward, step.
Everything else here is plumbing: building parts from a config, moving
batches to the device, evaluating and checkpointing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import optim as manual_optim
from .data import SpiralDataset, collate_dicts
from .layers import MLP
from .losses import cross_entropy


@dataclass
class TrainConfig:
    optimizer: str = "adamw"  # "manual-sgd", "sgd", "manual-adamw" or "adamw"
    lr: float = 1e-2
    momentum: float = 0.9  # SGD only
    weight_decay: float = 0.0
    epochs: int = 60
    batch_size: int = 64
    hidden: int = 64
    depth: int = 2
    dropout: float = 0.0
    max_grad_norm: float | None = None
    points_per_class: int = 300
    num_workers: int = 0
    seed: int = 0
    device: str = "cpu"


def build_optimizer(cfg: TrainConfig, params):
    params = list(params)
    if cfg.optimizer == "manual-sgd":
        return manual_optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "manual-adamw":
        return manual_optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def build(cfg: TrainConfig):
    """Data, model and optimizer, all seeded from cfg.seed."""
    torch.manual_seed(cfg.seed)
    train_ds = SpiralDataset(cfg.points_per_class, seed=cfg.seed)
    val_ds = SpiralDataset(cfg.points_per_class, seed=cfg.seed + 1)  # same spirals, fresh noise
    device = torch.device(cfg.device)
    # Input statistics come from the training split only, and live in buffers.
    model = MLP(2, cfg.hidden, train_ds.num_classes, cfg.depth, cfg.dropout,
                mean=train_ds.x.mean(0), std=train_ds.x.std(0)).to(device)
    optimizer = build_optimizer(cfg, model.parameters())
    shuffle_gen = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_dicts,
                              num_workers=cfg.num_workers, generator=shuffle_gen,
                              pin_memory=device.type == "cuda")  # the loader only pins for CUDA
    val_loader = DataLoader(val_ds, batch_size=512, collate_fn=collate_dicts)
    return model, optimizer, train_loader, val_loader, shuffle_gen


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    # non_blocking only overlaps the copy with compute when the source is pinned.
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_step(model, batch, optimizer, max_grad_norm: float | None = None) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)  # 1. forget the previous step's gradients
    logits = model(batch["x"])  # 2. forward: [B, 2] -> [B, C], records the graph
    loss = cross_entropy(logits, batch["y"])  # 3. scalar loss at the root of the graph
    loss.backward()  # 4. walk the graph backwards, write p.grad for every leaf
    if max_grad_norm is not None:
        norm = manual_optim.clip_grad_norm(model.parameters(), max_grad_norm)
    else:
        norm = manual_optim.grad_norm(model.parameters())
    optimizer.step()  # 5. read p.grad, update p in place
    return {"loss": loss.item(), "grad_norm": norm.item()}


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()  # dropout off; a model with batch norm would also switch to running stats
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch["x"])
        total_loss += cross_entropy(logits, batch["y"]).item() * len(batch["y"])
        correct += (logits.argmax(-1) == batch["y"]).sum().item()
        n += len(batch["y"])
    model.train()
    return {"val_loss": total_loss / n, "val_acc": correct / n}


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, cfg: TrainConfig) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "config": asdict(cfg), "rng": torch.get_rng_state()}, path)


def fit(
    cfg: TrainConfig,
    resume_from: str | Path | None = None,
    save_to: str | Path | None = None,
    on_epoch_end: Callable[[int, torch.nn.Module, dict], None] | None = None,
):
    """Train for cfg.epochs epochs (counting any already done in `resume_from`).

    Returns (model, history), where history has one dict per epoch.
    """
    model, optimizer, train_loader, val_loader, shuffle_gen = build(cfg)
    device = torch.device(cfg.device)
    start = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["rng"])
        start = ckpt["epoch"] + 1

    history = []
    model.train()
    for epoch in range(start, cfg.epochs):
        # Seeding the shuffle by epoch makes a resumed run see the same batches.
        shuffle_gen.manual_seed(cfg.seed * 10_000 + epoch)
        losses, norms = [], []
        for batch in train_loader:
            stats = train_step(model, to_device(batch, device), optimizer, cfg.max_grad_norm)
            losses.append(stats["loss"])
            norms.append(stats["grad_norm"])
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "grad_norm": max(norms),
               **evaluate(model, val_loader, device)}
        history.append(row)
        if on_epoch_end is not None:
            on_epoch_end(epoch, model, row)
        if save_to is not None:
            save_checkpoint(save_to, model, optimizer, epoch, cfg)
    return model, history
