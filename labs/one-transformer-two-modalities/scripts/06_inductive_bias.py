"""Section 9, part 1: the priors a CNN hard-codes and a ViT does not. Runs offline, at initialisation.

Translation equivariance: shift the image and a convolution's output shifts the same way, for any weights.
A ViT only gets this for shifts that are whole patches, and only without a position table. Locality: one
3x3 convolution sees 9 pixels; one attention layer sees the whole image.
"""

import _setup
import torch
from torch import nn

from twomod import BlockConfig, ViT

_setup.header()
torch.manual_seed(0)
images = torch.randn(4, 3, 32, 32)


def shift(x: torch.Tensor, pixels: int) -> torch.Tensor:
    return torch.roll(x, shifts=(pixels, pixels), dims=(2, 3))  # circular, so no border effects


_setup.section("A convolution is translation-equivariant by construction")
conv = nn.Conv2d(3, 16, 3, padding=1, padding_mode="circular")
with torch.no_grad():
    for s in (1, 3, 4):
        d = (conv(shift(images, s)) - shift(conv(images), s)).abs().max().item()
        print(f"conv(shift(x, {s})) vs shift(conv(x), {s}), max |difference|: {d:.1e}")

_setup.section("A ViT is not: only whole-patch shifts, only with no position table")
print("ViT, patch 4, mean pooling, random weights. Relative change of the logits when the image is shifted:")
for pos in ("none", "learned"):
    vit = ViT(BlockConfig(dim=64, n_heads=4), depth=2, pos=pos, pool="mean").eval()
    if pos == "learned":
        nn.init.normal_(vit.pos, std=1.0)
    with torch.no_grad():
        ref = vit(images)
        cells = []
        for s in (1, 3, 4, 8):
            cells.append(f"shift {s}: {((vit(shift(images, s)) - ref).norm() / ref.norm()).item():.1e}")
    print(f"  pos={pos:<8} " + "   ".join(cells))

_setup.section("Receptive field of the first layer")
x = torch.zeros(1, 3, 32, 32, requires_grad=True)
conv(x)[0, :, 16, 16].sum().backward()
print(f"3x3 conv: output pixel (16, 16) depends on {(x.grad.abs().sum(1) > 0).sum().item()} of 1,024 input pixels")
x = torch.zeros(1, 3, 32, 32, requires_grad=True)
vit = ViT(BlockConfig(dim=64, n_heads=4), depth=1, pos="learned", pool="mean")
h = vit.blocks[0](vit.tokens(x + torch.randn_like(x)), None)
h[0, 1 + 36].sum().backward()  # the token of one patch near the centre
print(f"one ViT block: the token of one patch depends on {(x.grad.abs().sum(1) > 0).sum().item()} of 1,024 pixels")
