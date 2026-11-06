"""CLIP's softmax loss against SigLIP's sigmoid loss, on the same similarity matrix.

    CLIP:   L = (CE(rows) + CE(columns)) / 2 over logits t * X Y^T. Every cell competes with its whole row.
    SigLIP: L = -(1/n) sum_ij log sigmoid(z_ij (t x_i . y_j + b)), z = +1 on the diagonal, -1 elsewhere.
            Each cell is its own binary problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def clip_loss(img: torch.Tensor, txt: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logits = scale * img @ txt.T
    target = torch.arange(len(img), device=img.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def siglip_loss(img: torch.Tensor, txt: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    logits = scale * img @ txt.T + bias
    z = 2 * torch.eye(len(img), device=img.device) - 1
    return -F.logsigmoid(z * logits).sum() / len(img)


def siglip_loss_chunked(img: torch.Tensor, txt: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor,
                        chunks: int) -> torch.Tensor:
    """The same loss computed one (image block, text block) pair at a time, as SigLIP does across devices.

    No block needs any other block's logits: there is no row normalisation to share. Each device keeps its
    images and passes text blocks around a ring; it never holds the full n x n matrix.
    """
    n = len(img)
    total = img.new_zeros(())
    for ib, ii in zip(img.chunk(chunks), torch.arange(n).chunk(chunks)):
        for tb, jj in zip(txt.chunk(chunks), torch.arange(n).chunk(chunks)):
            z = 2 * (ii[:, None] == jj[None, :]).float() - 1
            total = total - F.logsigmoid(z * (scale * ib @ tb.T + bias)).sum()
    return total / n


def logit_grads(logits: torch.Tensor, kind: str, bias: float = -10.0) -> torch.Tensor:
    """d loss / d logits for an n x n matrix of raw scores (already scaled)."""
    logits = logits.clone().requires_grad_(True)
    n = len(logits)
    if kind == "clip":
        t = torch.arange(n)
        loss = (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t)) / 2
    else:
        z = 2 * torch.eye(n) - 1
        loss = -F.logsigmoid(z * (logits + bias)).sum() / n
    (g,) = torch.autograd.grad(loss, logits)
    return g


def coupling(n: int = 8, seed: int = 0, cell: tuple[int, int] = (0, 1), delta: float = 3.0) -> dict[str, int]:
    """Raise one off-diagonal logit. How many other cells' gradients change? Softmax: the whole row and column."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, n, generator=g) * 2
    bumped = logits.clone()
    bumped[cell] += delta
    out = {}
    for kind in ("clip", "siglip"):
        diff = (logit_grads(bumped, kind) - logit_grads(logits, kind)).abs() > 1e-7
        diff[cell] = False
        out[kind] = int(diff.sum())
    return out


# A toy two-tower problem: one hidden latent, two noisy non-linear views of it ("image" and "text").
@dataclass
class ToyData:
    latent: int = 16
    d_img: int = 64
    d_txt: int = 48
    noise: float = 0.6
    seed: int = 0

    def __post_init__(self):
        g = torch.Generator().manual_seed(self.seed)
        self.a = torch.randn(self.latent, self.d_img, generator=g) / math.sqrt(self.latent)
        self.b = torch.randn(self.latent, self.d_txt, generator=g) / math.sqrt(self.latent)

    def sample(self, n: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(n, self.latent, generator=g)
        x = torch.tanh(z @ self.a) + self.noise * torch.randn(n, self.d_img, generator=g)
        y = torch.tanh(z @ self.b) + self.noise * torch.randn(n, self.d_txt, generator=g)
        return x, y


class Tower(nn.Module):
    def __init__(self, d_in: int, d_out: int = 32, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def retrieval_top1(fi: torch.Tensor, ft: torch.Tensor) -> float:
    """Image -> text top-1 among all held-out pairs."""
    return float(((fi @ ft.T).argmax(1) == torch.arange(len(fi))).float().mean())


def train(kind: str, batch: int, samples: int, data: ToyData, lr: float = 2e-3, seed: int = 0,
          eval_n: int = 2000) -> dict:
    """Train both towers for a fixed number of samples seen, so every batch size gets the same data budget.

    Initial temperature and bias follow the papers: CLIP t = 1/0.07; SigLIP t = 10, b = -10.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    fi, ft = Tower(data.d_img), Tower(data.d_txt)
    log_t = nn.Parameter(torch.tensor(math.log(1 / 0.07 if kind == "clip" else 10.0)))
    bias = nn.Parameter(torch.tensor(-10.0))
    params = [*fi.parameters(), *ft.parameters(), log_t] + ([bias] if kind == "siglip" else [])
    opt = torch.optim.Adam(params, lr=lr)
    steps = samples // batch
    for _ in range(steps):
        x, y = data.sample(batch, g)
        t = log_t.exp().clamp(max=100)
        loss = clip_loss(fi(x), ft(y), t) if kind == "clip" else siglip_loss(fi(x), ft(y), t, bias)
        opt.zero_grad()
        loss.backward()
        opt.step()
    ge = torch.Generator().manual_seed(12345)
    x, y = data.sample(eval_n, ge)
    with torch.no_grad():
        acc = retrieval_top1(fi(x), ft(y))
    return {"loss": kind, "batch": batch, "steps": steps, "top1": acc, "final_train_loss": float(loss.detach())}
