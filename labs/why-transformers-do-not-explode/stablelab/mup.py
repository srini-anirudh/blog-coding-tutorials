"""Part X of the post: standard parameterization (SP) versus the Maximal Update Parametrization (muP).

A three-matrix MLP, input -> hidden -> hidden -> output, trained with Adam. The two parameterizations
differ only in output-layer init and per-tensor learning rates, following Table 3 of Tensor Programs V
(Adam column), written relative to a base width so that SP and muP coincide at the base width:

                    SP                      muP
    input  W        init 1/fan_in, lr eta   init 1/fan_in, lr eta
    hidden W        init 1/fan_in, lr eta   init 1/fan_in, lr eta * base / width
    output W        init 1/fan_in, lr eta   init variance (1/fan_in) * (base / width), lr eta * base / width
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, d_in: int, width: int, d_out: int, param: str = "sp", base: int = 64) -> None:
        super().__init__()
        if param not in ("sp", "mup"):
            raise ValueError(param)
        self.param, self.width, self.base = param, width, base
        self.inp = nn.Linear(d_in, width)
        self.hid = nn.Linear(width, width)
        self.out = nn.Linear(width, d_out)
        for m in (self.inp, self.hid, self.out):
            nn.init.normal_(m.weight, std=1 / math.sqrt(m.in_features))
            nn.init.zeros_(m.bias)
        if param == "mup":
            self.out.weight.data.mul_(math.sqrt(base / width))  # variance ~ 1/width^2

    def forward(self, x: torch.Tensor, keep: dict | None = None) -> torch.Tensor:
        h1 = F.relu(self.inp(x))
        h2 = F.relu(self.hid(h1))
        out = self.out(h2)
        if keep is not None:
            keep.update(h1=h1, h2=h2, out=out)
        return out

    def param_groups(self, lr: float) -> list[dict]:
        if self.param == "sp":
            return [{"params": self.parameters(), "lr": lr}]
        r = self.base / self.width
        return [
            {"params": [*self.inp.parameters(), self.hid.bias, self.out.bias], "lr": lr},
            {"params": [self.hid.weight, self.out.weight], "lr": lr * r},
        ]


def context_dataset(ids: torch.Tensor, vocab: int, context: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-character prediction from the previous `context` characters, one-hot and concatenated."""
    n = len(ids) - context
    idx = torch.arange(context)[None, :] + torch.arange(n)[:, None]
    x = F.one_hot(ids[idx], vocab).float().reshape(n, -1)
    return x, ids[context:]


def train_mlp(model: MLP, x: torch.Tensor, y: torch.Tensor, lr: float, steps: int, batch: int = 256,
              seed: int = 0) -> float:
    """Adam, constant LR. Returns the mean training loss over the last 20% of steps (inf if it diverged)."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.param_groups(lr))
    tail = []
    for step in range(steps):
        i = torch.randint(len(x), (batch,), generator=gen)
        loss = F.cross_entropy(model(x[i]), y[i])
        if not torch.isfinite(loss):
            return float("inf")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step >= int(0.8 * steps):
            tail.append(loss.item())
    return sum(tail) / len(tail)


def coordinate_check(width: int, param: str, x: torch.Tensor, y: torch.Tensor, lr: float = 1e-2,
                     steps: int = 5, seed: int = 0) -> dict:
    """RMS of each layer's output after a few Adam steps, minus its value at init. Flat in width under muP."""
    torch.manual_seed(seed)
    model = MLP(x.shape[1], width, int(y.max()) + 1, param)
    probe = x[:512]
    before, after = {}, {}
    model(probe, before)
    train_mlp(model, x, y, lr, steps, seed=seed)
    model(probe, after)
    return {k: (after[k] - before[k]).pow(2).mean().sqrt().item() for k in before}
