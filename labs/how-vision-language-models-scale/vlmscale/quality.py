"""A small real version of the quality-quantity trade-off (Goyal et al., 2024).

A fixed random "teacher" MLP labels points in R^d. Every pool sample gets a quality score q in
[0, 1], and its label is replaced by a random class with probability noise_max * (1 - q). We then
filter the pool to its top fraction by q and train a fresh student on the kept samples for a budget
of sample views, cycling through the kept set (so a small kept set is repeated many times).
Test accuracy is measured on clean labels.

Nothing here is a contrastive image-text model. It reproduces the mechanism only: clean data that
runs out and repeats, against noisy data that stays fresh.
"""

from __future__ import annotations

import torch
from torch import nn


def make_pool(n: int, dim: int = 24, classes: int = 10, noise_max: float = 0.8, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    teacher = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, classes))
    with torch.no_grad():
        for p in teacher.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * (1.5 / p.shape[-1] ** 0.5 if p.dim() > 1 else 0.1))
        x = torch.randn(n, dim, generator=gen)
        clean = teacher(x).argmax(-1)
        x_test = torch.randn(20_000, dim, generator=gen)
        y_test = teacher(x_test).argmax(-1)
    q = torch.rand(n, generator=gen)
    flip = torch.rand(n, generator=gen) < noise_max * (1 - q)
    noisy = torch.where(flip, torch.randint(classes, (n,), generator=gen), clean)
    return {"x": x, "y": noisy, "clean": clean, "q": q, "x_test": x_test, "y_test": y_test, "classes": classes}


def filter_top(pool: dict, fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    k = int(round(fraction * len(pool["q"])))
    idx = pool["q"].argsort(descending=True)[:k]
    return pool["x"][idx], pool["y"][idx]


def train_student(x, y, x_test, y_test, classes: int, views: list[int], batch: int = 256, lr: float = 1e-3,
                  width: int = 256, seed: int = 0) -> list[dict]:
    """Train once with a constant learning rate and report clean test accuracy at each view budget."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                          nn.Linear(width, classes))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed + 7)
    order = torch.randperm(len(x), generator=gen)
    out, seen, cursor, targets = [], 0, 0, sorted(views)
    while targets:
        if cursor + batch > len(order):
            order, cursor = torch.randperm(len(x), generator=gen), 0
        idx = order[cursor:cursor + batch]
        cursor += batch
        loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        seen += len(idx)
        while targets and seen >= targets[0]:
            with torch.no_grad():
                acc = float((model(x_test).argmax(-1) == y_test).float().mean())
            out.append({"views": targets.pop(0), "epochs": seen / len(x), "test_acc": acc})
    return out
