"""What a representation keeps: meaning or pixels.

Three encoders with the same architecture (the tokenizer's conv encoder, 16 x 16 -> 4 x 4 x 64) and three
training signals:

- pixel:    reconstruct the image through a decoder (a VQGAN-style objective, without the quantizer)
- semantic: match the caption, which names only the shape (a CLIP-style objective over the 4 captions)
- both:     the sum of the two losses (the "unified tokenizer" idea of VILA-U and UniTok)

Each frozen encoder is then scored two ways. Reading: a linear probe for the shape, fitted on few labels.
Drawing: a fresh decoder trained to rebuild the pixels from the frozen features.

`Judge` is a separate small CNN classifier used to grade generated images, the way GenEval grades with a
detector.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .tokenizers import Tokenizer


def conv_encoder(dim: int = 64) -> nn.Module:
    return Tokenizer(nn.Identity(), dim).enc


def conv_decoder(dim: int = 64) -> nn.Module:
    return Tokenizer(nn.Identity(), dim).dec


class CaptionHead(nn.Module):
    """Pool the feature map, project, and score against one learned embedding per caption."""

    def __init__(self, dim: int = 64, n_captions: int = 4, temperature: float = 0.07):
        super().__init__()
        self.proj = nn.Linear(dim, 64)
        self.captions = nn.Parameter(torch.randn(n_captions, 64))
        self.temperature = temperature

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        img = F.normalize(self.proj(feats.mean(dim=(2, 3))), dim=-1)
        txt = F.normalize(self.captions, dim=-1)
        return img @ txt.t() / self.temperature


def train_encoder(objective: str, images: torch.Tensor, shape: torch.Tensor, steps: int, batch: int = 128,
                  lr: float = 2e-3, seed: int = 0) -> nn.Module:
    assert objective in ("pixel", "semantic", "both")
    torch.manual_seed(seed)
    enc, dec, head = conv_encoder(), conv_decoder(), CaptionHead()
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters(), *head.parameters()], lr=lr)
    for _ in range(steps):
        i = torch.randint(0, len(images), (batch,))
        feats = enc(images[i])
        loss = torch.zeros(())
        if objective in ("pixel", "both"):
            loss = loss + F.mse_loss(dec(feats), images[i]) * 20  # scale so the two losses start comparable
        if objective in ("semantic", "both"):
            loss = loss + F.cross_entropy(head(feats), shape[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    return enc


@torch.no_grad()
def features(enc: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return enc(images)


def linear_probe(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor,
                 n_classes: int, steps: int = 500, wd: float = 1e-3) -> float:
    """Standardise the features, fit logistic regression with full-batch Adam, return test accuracy."""
    torch.manual_seed(0)
    tx, vx = train_x.flatten(1), test_x.flatten(1)
    mu, sd = tx.mean(0), tx.std(0) + 1e-5
    tx, vx = (tx - mu) / sd, (vx - mu) / sd
    lin = nn.Linear(tx.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        loss = F.cross_entropy(lin(tx), train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((lin(vx).argmax(-1) == test_y).float().mean())


def decoder_probe(train_f: torch.Tensor, train_img: torch.Tensor, test_f: torch.Tensor, test_img: torch.Tensor,
                  steps: int, batch: int = 128, seed: int = 0) -> float:
    """Train a fresh decoder on frozen features. Return test reconstruction MSE."""
    torch.manual_seed(seed)
    dec = conv_decoder()
    opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
    for _ in range(steps):
        i = torch.randint(0, len(train_f), (batch,))
        loss = F.mse_loss(dec(train_f[i]), train_img[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(F.mse_loss(dec(test_f), test_img))


class Judge(nn.Module):
    """Small CNN that reads a 16 x 16 image and predicts the shape and the side."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU(),
        )
        self.shape = nn.Linear(128, 4)
        self.side = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return self.shape(h), self.side(h)


def train_judge(data: dict, steps: int = 1500, batch: int = 128, seed: int = 0) -> Judge:
    torch.manual_seed(seed)
    judge = Judge()
    opt = torch.optim.Adam(judge.parameters(), lr=1e-3)
    for _ in range(steps):
        i = torch.randint(0, len(data["images"]), (batch,))
        ls, lsd = judge(data["images"][i])
        loss = F.cross_entropy(ls, data["shape"][i]) + F.cross_entropy(lsd, data["side"][i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    judge.eval()
    return judge


@torch.no_grad()
def judge_accuracy(judge: Judge, images: torch.Tensor, shape: torch.Tensor, side: torch.Tensor | None = None) -> dict:
    ls, lsd = judge(images)
    out = {"shape_acc": float((ls.argmax(-1) == shape).float().mean())}
    if side is not None:
        out["side_acc"] = float((lsd.argmax(-1) == side).float().mean())
        out["both_acc"] = float(((ls.argmax(-1) == shape) & (lsd.argmax(-1) == side)).float().mean())
    return out
