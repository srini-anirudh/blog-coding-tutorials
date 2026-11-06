"""Attention over patches: global, windowed, shifted windows, and 2D positions."""

import _setup  # noqa: F401
import torch

from attnzoo.attention_naive import naive_attention
from attnzoo.masks import render
from attnzoo.pos2d import RelativePositionBias2D, Rope2D, sincos_2d_embedding
from attnzoo.vision import WindowAttention, patchify, shifted_window_mask, window_merge, window_partition

torch.manual_seed(0)
# A synthetic image: dark background, one bright square. Height and width 64, patches of 8.
img = torch.zeros(1, 3, 64, 64) + 0.05 * torch.randn(1, 3, 64, 64)
img[:, :, 16:40, 24:56] += 1.0
tokens = patchify(img, 8)
print("image", tuple(img.shape), "-> patches", tuple(tokens.shape))

# Use pixels directly as q, k, v (no learned projection), so the map only reflects content.
x = tokens[:, None]  # (B, 1, N, 192)
_, w = naive_attention(x, x, x, scale=3.0, return_weights=True)
print("\npatch-to-patch weights", tuple(w.shape), "(no causal mask: every patch sees all 64)")
for label, query in [("inside the square", 3 * 8 + 5), ("in the background", 0)]:
    row = w[0, 0, query].view(8, 8)
    print(f"query patch {query} ({label}): weight on each patch, x 100")
    print("\n".join(" ".join(f"{int(v * 100):>2}" for v in r) for r in row))

print("\nWindow partition (B=1, H=W=8, C=32, window 4):")
grid = torch.randn(1, 8, 8, 32)
win = window_partition(grid, 4)
print("  grid", tuple(grid.shape), "-> windows", tuple(win.shape), "-> merged", tuple(window_merge(win, 4, 8, 8).shape))
print(f"  scores per layer: global {64 ** 2:,}, windowed {4 * 16 ** 2:,}")

print("\nA token's reach after each layer (non-zero change in the output when token (0,0) changes):")
attn_plain, attn_shift = WindowAttention(32, 4, window=4).eval(), WindowAttention(32, 4, window=4, shift=2).eval()
y = grid.clone()
y[:, 0, 0] += 1.0
plain = (attn_plain(grid) - attn_plain(y)).abs().sum(-1)[0] > 0
print("after one plain window layer:\n" + render(plain))
two = (attn_shift(attn_plain(grid)) - attn_shift(attn_plain(y))).abs().sum(-1)[0] > 1e-6
print("after plain, then shifted:\n" + render(two))

print("\nShifted-window mask of the bottom-right window (8 x 8 grid, window 4, shift 2):")
print(render(shifted_window_mask(8, 8, 4, 2)[-1]))

print("\n2D positions:")
print("  sincos_2d_embedding(8, 8, 32):", tuple(sincos_2d_embedding(8, 8, 32).shape))
print("  RelativePositionBias2D(window=4, heads=4):", tuple(RelativePositionBias2D(4, 4)().shape), "from a table of", tuple(RelativePositionBias2D(4, 4).table.shape))
rope = Rope2D(16, 32)
q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)


def score(qr: int, qc: int, kr: int, kc: int) -> float:
    return (rope(q, torch.tensor([qr]), torch.tensor([qc])) * rope(k, torch.tensor([kr]), torch.tensor([kc]))).sum().item()


print(f"  2D RoPE score (5,3)->(2,1): {score(5, 3, 2, 1):.4f}")
print(f"  2D RoPE score (15,13)->(12,11), same offset: {score(15, 13, 12, 11):.4f}")
print(f"  2D RoPE score (5,3)->(1,2), different offset: {score(5, 3, 1, 2):.4f}")
