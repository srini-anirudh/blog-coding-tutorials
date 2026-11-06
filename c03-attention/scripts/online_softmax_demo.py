"""Walk the online softmax through two tiles, printing the running m and l, then check it against naive."""

import math

import _setup  # noqa: F401
import torch

from attnzoo.attention_naive import naive_attention
from attnzoo.attention_tiled import tiled_attention

torch.manual_seed(3)
d = 4
q = torch.randn(1, 1, 1, d)  # one query
k = torch.randn(1, 1, 6, d) * 2  # six keys, in two tiles of three
v = torch.randn(1, 1, 6, 2)
scores = ((q @ k.transpose(-2, -1)) / math.sqrt(d))[0, 0, 0]
print("all six scores:", scores)
print("true softmax:  ", torch.softmax(scores, -1))

m, l, acc = torch.tensor(float("-inf")), torch.tensor(0.0), torch.zeros(2)
for tile, (s, e) in enumerate([(0, 3), (3, 6)]):
    tile_scores = scores[s:e]
    m_new = torch.maximum(m, tile_scores.max())
    alpha = torch.exp(m - m_new)
    p = torch.exp(tile_scores - m_new)
    l = alpha * l + p.sum()
    acc = alpha * acc + p @ v[0, 0, s:e]
    print(f"\ntile {tile}: scores {tile_scores}")
    print(f"  m: {m.item():.3f} -> {m_new.item():.3f}   alpha = exp(m - m_new) = {alpha.item():.3f}")
    print(f"  p = exp(s - m_new): {p}")
    print(f"  l = {l.item():.3f}   acc = {acc}")
    m = m_new

print("\nacc / l:          ", acc / l)
print("softmax @ v:      ", torch.softmax(scores, -1) @ v[0, 0])

print("\nFull function, causal, N = 512, tiles 64 x 64:")
q, k, v = torch.randn(2, 8, 512, 32), torch.randn(2, 2, 512, 32), torch.randn(2, 2, 512, 32)
trace: list[dict] = []
out = tiled_attention(q, k, v, causal=True, block_q=64, block_k=64, trace=trace)
ref = naive_attention(q, k, v, causal=True)
skipped = sum(t["skipped"] for t in trace)
print(f"max |tiled - naive| = {(out - ref).abs().max().item():.2e}")
print(f"tiles: {len(trace)} total, {skipped} skipped because they lie wholly in the future")
print(f"largest score tensor held: 64 x 64 per head = {64 * 64:,} entries, against {512 * 512:,} for naive")
