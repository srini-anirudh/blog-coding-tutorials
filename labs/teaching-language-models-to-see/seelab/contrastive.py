"""Contrastive pretraining: the CLIP softmax loss, the SigLIP sigmoid loss, and the modality gap.

u: B x d normalised image vectors, v: B x d normalised text vectors, s = u v^T is the B x B similarity matrix.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def clip_loss(u: torch.Tensor, v: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    """Softmax over every row (image -> caption) and every column (caption -> image), averaged."""
    logits = u @ v.T / tau
    target = torch.arange(u.shape[0], device=u.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def siglip_loss(u: torch.Tensor, v: torch.Tensor, t: float | torch.Tensor, b: float | torch.Tensor) -> torch.Tensor:
    """An independent yes-or-no decision for every cell: +1 on the diagonal, -1 elsewhere. Sum over cells / B."""
    logits = t * (u @ v.T) + b
    z = 2 * torch.eye(u.shape[0], device=u.device) - 1
    return -F.logsigmoid(z * logits).sum() / u.shape[0]


def siglip_loss_chunked(u, v, t, b, chunk: int) -> torch.Tensor:
    """The same loss, one block of the matrix at a time, as if each block lived on another device.

    Each block needs only its own rows of u and columns of v. Nothing is normalised across blocks.
    """
    total, n = u.new_zeros(()), u.shape[0]
    for i in range(0, n, chunk):
        for j in range(0, n, chunk):
            logits = t * (u[i:i + chunk] @ v[j:j + chunk].T) + b
            rows = torch.arange(i, min(i + chunk, n), device=u.device)[:, None]
            cols = torch.arange(j, min(j + chunk, n), device=u.device)[None, :]
            z = torch.where(rows == cols, 1.0, -1.0)
            total = total + (-F.logsigmoid(z * logits)).sum()
    return total / n


def clip_loss_chunked_wrong(u, v, tau, chunk: int) -> torch.Tensor:
    """What goes wrong if CLIP is split the same way: each softmax only sees the captions in its own block."""
    parts = [clip_loss(u[i:i + chunk], v[i:i + chunk], tau) for i in range(0, u.shape[0], chunk)]
    return torch.stack(parts).mean()


def modality_gap(u: torch.Tensor, v: torch.Tensor) -> dict[str, float]:
    """Liang et al. (2022): distance between the two modality centroids, plus how narrow each cone is."""
    u, v = F.normalize(u, dim=-1), F.normalize(v, dim=-1)

    def cone(x):  # mean cosine between distinct vectors of one modality; 0 for random directions
        n = x.shape[0]
        sim = x @ x.T
        return ((sim.sum() - sim.diagonal().sum()) / (n * (n - 1))).item()

    return {
        "gap": (u.mean(0) - v.mean(0)).norm().item(),
        "cone_image": cone(u),
        "cone_text": cone(v),
        "cos_matched": (u * v).sum(-1).mean().item(),
        "cos_image_text_all": (u @ v.T).mean().item(),
    }


# ---- a toy two-tower model on synthetic paired data ------------------------------------------------------------
class ToyPairs:
    """Paired "images" and "captions" that share a hidden cause z, seen through two different random maps."""

    def __init__(self, d_z: int = 16, d_img: int = 64, d_txt: int = 32, noise: float = 0.3, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.d_z, self.noise = d_z, noise
        self.a_img = torch.randn(d_z, d_img, generator=g) / math.sqrt(d_z)
        self.a_txt = torch.randn(d_z, d_txt, generator=g) / math.sqrt(d_z)

    def sample(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(n, self.d_z, generator=generator)
        img = torch.tanh(z @ self.a_img) + self.noise * torch.randn(n, self.a_img.shape[1], generator=generator)
        txt = torch.tanh(z @ self.a_txt) + self.noise * torch.randn(n, self.a_txt.shape[1], generator=generator)
        return img, txt


def tower(d_in: int, d_out: int = 32, width: int = 128, depth: int = 4) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(depth - 1):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class TwoTower(nn.Module):
    def __init__(self, d_img: int, d_txt: int, d_out: int = 32, loss: str = "siglip"):
        super().__init__()
        self.image, self.text, self.loss_name = tower(d_img, d_out), tower(d_txt, d_out), loss
        if loss == "siglip":  # the paper's initialisation: t = 10, b = -10
            self.log_t, self.b = nn.Parameter(torch.tensor(math.log(10.0))), nn.Parameter(torch.tensor(-10.0))
        else:  # CLIP: tau = 0.07
            self.log_t, self.b = nn.Parameter(torch.tensor(math.log(1 / 0.07))), None

    def embed(self, img, txt):
        return F.normalize(self.image(img), dim=-1), F.normalize(self.text(txt), dim=-1)

    def forward(self, img, txt):
        u, v = self.embed(img, txt)
        t = self.log_t.exp()
        return siglip_loss(u, v, t, self.b) if self.loss_name == "siglip" else clip_loss(u, v, 1 / t)


@torch.no_grad()
def retrieval_top1(model: TwoTower, img, txt) -> float:
    """Image -> caption retrieval among all held-out captions."""
    u, v = model.embed(img, txt)
    return ((u @ v.T).argmax(1) == torch.arange(len(u))).float().mean().item()


def train_toy(loss: str, batch_size: int, steps: int, data: ToyPairs, seed: int = 0, lr: float = 1e-3,
              log_every: int = 0) -> tuple[TwoTower, list[dict]]:
    torch.manual_seed(seed)
    model = TwoTower(data.a_img.shape[1], data.a_txt.shape[1], loss=loss)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed + 1)
    history = []
    for step in range(steps):
        img, txt = data.sample(batch_size, g)
        loss_value = model(img, txt)
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            history.append({"step": step, "loss": loss_value.item()})
    return model, history
