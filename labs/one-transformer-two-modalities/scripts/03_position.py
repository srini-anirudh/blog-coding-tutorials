"""Section 4: position, the only geometry.

Attention sees a set: shuffle the input rows and the output rows shuffle the same way. A ViT with no position
embedding cannot tell a shuffled image from the original. Then the two lineages: sinusoids (a fixed rotation per
offset), RoPE in 1D and 2D (imported from C3's attnzoo), and the interpolation step a learned 2D table needs at
a new resolution. Runs offline.
"""

import sys

import _setup
import torch

from twomod import Block, BlockConfig, ViT
from twomod.front import patchify, unpatchify
from twomod.positions import (
    alignment_score,
    interpolate_table,
    neighbour_score,
    sincos_2d,
    sinusoidal,
    sinusoidal_shift_matrix,
)

sys.path.insert(0, str(_setup.C3))
from attnzoo.pos2d import Rope2D  # noqa: E402
from attnzoo.rope import RotaryEmbedding, rotate  # noqa: E402

_setup.header()
torch.manual_seed(0)

_setup.section("Attention is permutation-equivariant")
block = Block(BlockConfig(dim=64, n_heads=4)).eval()
x = torch.randn(1, 16, 64)
perm = torch.randperm(16)
with torch.no_grad():
    diff = (block(x[:, perm]) - block(x)[:, perm]).abs().max().item()
print(f"block(x[perm]) vs block(x)[perm], no mask, max |difference|: {diff:.1e}")

images = torch.randn(4, 3, 32, 32)
patch_perm = torch.randperm(64)
shuffled = unpatchify(patchify(images, 4)[:, patch_perm], 4, 3, 32, 32)  # the same tiles, in a new order
for pos in ("none", "learned", "sincos2d"):
    vit = ViT(BlockConfig(dim=64, n_heads=4), depth=2, pos=pos).eval()
    if pos == "learned":
        torch.nn.init.normal_(vit.pos, std=1.0)  # a trained table is far from zero; use a visible one here
    with torch.no_grad():
        d = (vit(images) - vit(shuffled)).abs().max().item()
    print(f"ViT pos={pos:<9} logits for the image vs its patch-shuffled copy, max |difference|: {d:.1e}")

_setup.section("Sinusoids: every offset is one fixed rotation")
table = sinusoidal(512, 64)
worst = max(
    (table[k:] - table[: 512 - k] @ sinusoidal_shift_matrix(k, 64)).abs().max().item() for k in (1, 7, 100)
)
print(f"PE(t + k) vs PE(t) @ M_k for k in (1, 7, 100), all t, max |difference|: {worst:.1e}")

_setup.section("RoPE (C3 attnzoo): the score depends only on the offset")
rope = RotaryEmbedding(64, max_seq_len=4096)
q, k = torch.randn(1, 1, 1, 64), torch.randn(1, 1, 1, 64)


def score(m: int, n: int) -> float:
    cq, sq = rope(torch.tensor([[m]]))
    ck, sk = rope(torch.tensor([[n]]))
    return (rotate(q, cq, sq) * rotate(k, ck, sk)).sum().item()


base = score(10, 3)
shifted = [score(10 + s, 3 + s) for s in (1, 100, 2000)]
print(f"score(m=10, n=3) = {base:.5f}; the same pair shifted by 1, 100, 2000: "
      f"{', '.join(f'{v:.5f}' for v in shifted)}; max drift {max(abs(v - base) for v in shifted):.1e} "
      f"(float32 rounding of large angles)")
rope2d = Rope2D(64, max_side=64)
q2, k2 = torch.randn(1, 1, 1, 64), torch.randn(1, 1, 1, 64)


def score2d(qr, qc, kr, kc) -> float:
    rq = rope2d(q2, torch.tensor([qr]), torch.tensor([qc]))
    rk = rope2d(k2, torch.tensor([kr]), torch.tensor([kc]))
    return (rq * rk).sum().item()


b2 = score2d(5, 9, 2, 4)
drift2 = max(abs(score2d(5 + a, 9 + b, 2 + a, 4 + b) - b2) for a, b in [(1, 0), (0, 7), (20, 30)])
print(f"2D RoPE: score((5,9), (2,4)) = {b2:.5f}; shifted in rows and columns, max drift {drift2:.1e}")

_setup.section("A learned 2D table at a new resolution must be interpolated")
old, new = (14, 14), (24, 24)
learned_like = sincos_2d(*old, 128) + 0.05 * torch.randn(196, 128)  # a smooth stand-in for a trained table
random_table = torch.randn(196, 128)
print(f"a 1D table trained at 14 x 14 has 196 rows; 24 x 24 needs {new[0] * new[1]} positions")
big = interpolate_table(learned_like, old, new)
naive = learned_like.repeat(3, 1)[:576]  # reuse the 196 rows in raster order instead
print(f"neighbour score (closest row is a grid neighbour): random table {neighbour_score(random_table, *old):.3f}, "
      f"smooth 14 x 14 table {neighbour_score(learned_like, *old):.3f}")
print("alignment (each 24 x 24 cell looks most like the 14 x 14 cell it covers):")
print(f"  bicubic interpolation {alignment_score(learned_like, big, old, new):.3f}; "
      f"reusing raster rows {alignment_score(learned_like, naive, old, new):.3f}")
print("results/vit_vs_cnn.json repeats both scores on a ViT trained on CIFAR-10 (script 07)")
