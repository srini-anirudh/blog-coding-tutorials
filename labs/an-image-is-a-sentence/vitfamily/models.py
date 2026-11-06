"""Two small classifiers for 32 x 32 images: a ViT with the post's options, and a CNN.

TinyViT options
  readout="cls" | "gap"   learned class token, or global average pool over patch tokens
  distill=True            DeiT: a second special token with its own head, trained on the teacher's labels
  gpsa_layers=k           ConViT: the first k blocks use gated positional self-attention on patches only;
                          the special tokens join after them, as in ConViT
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import GPSA
from .blocks import ViTBlock
from .patch import PatchEmbed


class TinyViT(nn.Module):
    def __init__(
        self,
        img: int = 32,
        patch: int = 4,
        dim: int = 64,
        depth: int = 6,
        heads: int = 4,
        num_classes: int = 10,
        readout: str = "cls",
        distill: bool = False,
        gpsa_layers: int = 0,
    ) -> None:
        super().__init__()
        if readout not in ("cls", "gap"):
            raise ValueError("readout is 'cls' or 'gap'")
        if distill and readout != "cls":
            raise ValueError("the distillation token sits next to a class token")
        self.readout, self.distill = readout, distill
        side = img // patch
        self.embed = PatchEmbed(patch, dim)
        self.pos = nn.Parameter(torch.randn(1, side * side, dim) * 0.02)  # learned 1D absolute
        n_special = (readout == "cls") + distill
        self.special = nn.Parameter(torch.randn(1, n_special, dim) * 0.02) if n_special else None
        self.blocks = nn.ModuleList(
            [ViTBlock(dim, heads, GPSA(dim, heads, side) if i < gpsa_layers else None) for i in range(depth)]
        )
        self.gpsa_layers = gpsa_layers
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.head_dist = nn.Linear(dim, num_classes) if distill else None

    def features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.embed(images) + self.pos
        for i, blk in enumerate(self.blocks):
            if i == self.gpsa_layers and self.special is not None:
                x = torch.cat([self.special.expand(x.size(0), -1, -1), x], dim=1)
            x = blk(x)
        if self.gpsa_layers == len(self.blocks) and self.special is not None:
            x = torch.cat([self.special.expand(x.size(0), -1, -1), x], dim=1)
        return self.norm(x)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(images)
        if self.readout == "gap":
            return {"logits": self.head(x.mean(dim=1))}
        out = {"cls_feat": x[:, 0], "logits_cls": self.head(x[:, 0])}
        if self.distill:
            out["dist_feat"] = x[:, 1]
            out["logits_dist"] = self.head_dist(x[:, 1])
            out["logits"] = (out["logits_cls"] + out["logits_dist"]) / 2  # DeiT inference: average the heads
        else:
            out["logits"] = out["logits_cls"]
        return out


def _conv_bn(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.a = _conv_bn(c, c)
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.b(self.a(x)))


class TinyCNN(nn.Module):
    """Locality, weight sharing and a 32 -> 16 -> 8 hierarchy, built in before any data is seen."""

    def __init__(self, num_classes: int = 10, width: int = 32) -> None:
        super().__init__()
        w = width
        self.body = nn.Sequential(
            _conv_bn(3, w), ResBlock(w),
            _conv_bn(w, 2 * w, 2), ResBlock(2 * w),
            _conv_bn(2 * w, 4 * w, 2), ResBlock(4 * w),
        )  # fmt: skip
        self.head = nn.Linear(4 * w, num_classes)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.head(self.body(images).mean(dim=(2, 3)))}


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
