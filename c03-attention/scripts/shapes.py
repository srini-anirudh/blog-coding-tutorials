"""One head, then many heads, then cross-attention, printing every tensor shape."""

import _setup  # noqa: F401
import torch

from attnzoo.config import AttentionConfig
from attnzoo.core import scaled_dot_product_attention
from attnzoo.cross import CrossAttention
from attnzoo.heads import merge_heads, split_heads
from attnzoo.masks import causal_mask
from attnzoo.mha import FusedQKVAttention, MultiHeadAttention


def show(name: str, t: torch.Tensor) -> None:
    print(f"{name:<10} {str(tuple(t.shape)):<20} {t.dtype}")


torch.manual_seed(0)
print("--- attention on raw q, k, v (B=1, Nq=2, Nk=3, d=4) ---")
q, k, v = torch.randn(1, 2, 4), torch.randn(1, 3, 4), torch.randn(1, 3, 4)
out, w = scaled_dot_product_attention(q, k, v)
show("q", q), show("k", k), show("v", v), show("weights", w), show("out", out)
print("weights row sums:", w.sum(-1))
print("weights:\n", w[0])

print("\n--- multi-head reshapes (B=2, N=6, D=32, H=4) ---")
x = torch.randn(2, 6, 32)
mha = MultiHeadAttention(32, 4, causal=True)
projected = mha.q_proj(x)
show("x", x), show("q_proj(x)", projected)
q = split_heads(projected, 4)
show("q split", q)
k, v = split_heads(mha.k_proj(x), 4), split_heads(mha.v_proj(x), 4)
out, w = scaled_dot_product_attention(q, k, v, mask=causal_mask(6))
show("weights", w), show("out", out)
merged = merge_heads(out)
show("merged", merged), show("o_proj", mha.o_proj(merged))

print("\n--- three Linear layers against one fused Linear ---")
fused = FusedQKVAttention.from_separate(mha)
print("separate params:", sum(p.numel() for p in mha.parameters()))
print("fused params:   ", sum(p.numel() for p in fused.parameters()))
print("max |separate - fused|:", (mha(x) - fused(x)).abs().max().item())

print("\n--- cross-attention: the output length follows the queries ---")
cfg = AttentionConfig("cross", 32, 4, causal=False, positional_encoding="none", context_size=48)
cross = CrossAttention(cfg)
for name, nq, ns in [("text -> text", 6, 9), ("text -> image", 6, 196), ("latents -> image", 8, 729)]:
    print(f"{name:<18} query (2, {nq}, 32), context (2, {ns}, 48) -> {tuple(cross(torch.randn(2, nq, 32), torch.randn(2, ns, 48)).shape)}")
