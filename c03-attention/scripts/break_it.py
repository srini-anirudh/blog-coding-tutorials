"""Six ways to get attention wrong, and what each one looks like. Every failure here is real."""

import _setup  # noqa: F401
import torch
import torch.nn.functional as F

from attnzoo.config import AttentionConfig
from attnzoo.core import scaled_dot_product_attention
from attnzoo.gqa import GroupedQueryAttention
from attnzoo.kv_cache import KVCache
from attnzoo.masks import causal_mask

torch.manual_seed(0)

print("1. Missing 1/sqrt(d): softmax collapses as d grows")
print("      d   right max p   wrong max p")
for d in [8, 64, 512]:
    q, k, v = torch.randn(1, 1, 64, d), torch.randn(1, 1, 64, d), torch.randn(1, 1, 64, d)
    _, good = scaled_dot_product_attention(q, k, v)
    _, bad = scaled_dot_product_attention(q, k, v, scale=1.0)
    print(f"  {d:>5}{good.amax(-1).mean().item():>14.3f}{bad.amax(-1).mean().item():>14.3f}")

print("\n2. Mask inverted (triu instead of tril): every token reads only the future")
q, k, v = (torch.randn(1, 1, 6, 8) for _ in range(3))
wrong = torch.ones(6, 6, dtype=torch.bool).triu()
_, w = scaled_dot_product_attention(q, k, v, mask=wrong)
print("  weights of token 5 on keys 0..5:", w[0, 0, 5])
print("  weights of token 0 on keys 0..5:", w[0, 0, 0])
x, y = v.clone(), v.clone()
y[:, :, 5] += 1.0
leak = (scaled_dot_product_attention(q, k, x, mask=wrong)[0][:, :, 0] - scaled_dot_product_attention(q, k, y, mask=wrong)[0][:, :, 0]).abs().max()
print(f"  the causality check catches it: token 0 moved by {leak.item():.3f} when token 5 changed")

print("\n3. A row with no visible key: softmax of all -inf is NaN")
mask = causal_mask(4).clone()
mask[0, 0] = False  # e.g. left padding plus a causal mask on the first position
_, w = scaled_dot_product_attention(*(torch.randn(1, 1, 4, 8) for _ in range(3)), mask=mask)
print("  core.scaled_dot_product_attention row 0:", w[0, 0, 0])
print("  torch SDPA row 0 output:", F.scaled_dot_product_attention(*(torch.randn(1, 1, 4, 8) for _ in range(3)), attn_mask=mask)[0, 0, 0, :3])

print("\n4. is_causal=True while decoding: aligned top-left, so the new token sees only key 0")
q, k, v = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 5, 8), torch.randn(1, 1, 5, 8)
wrong = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("  max |output - v[0]| =", (wrong[0, 0, 0] - v[0, 0, 0]).abs().max().item())
print("  causal_mask(1, 5):", causal_mask(1, 5).int().tolist(), "against what is_causal builds:", torch.ones(1, 5, dtype=torch.bool).tril().int().tolist())

print("\n5. Forgetting the position offset while decoding: RoPE restarts at 0")
cfg = AttentionConfig("gqa", 64, 8, num_kv_heads=2, max_seq_len=64)
attn = GroupedQueryAttention(cfg).eval()
x = torch.randn(1, 12, 64)
full = attn(x)
cache = KVCache()
attn(x[:, :8], kv_cache=cache)
right = attn(x[:, 8:9], kv_cache=cache)
cache = KVCache()
attn(x[:, :8], kv_cache=cache)
wrong = attn(x[:, 8:9], kv_cache=cache, position_ids=torch.tensor([[0]]))  # the bug
print(f"  max error with position 8: {(right - full[:, 8:9]).abs().max().item():.1e}")
print(f"  max error with position 0: {(wrong - full[:, 8:9]).abs().max().item():.3f}")

print("\n6. Shape mistakes that fail loudly, which is the good kind")
out = torch.randn(2, 4, 5, 8)
try:
    out.transpose(1, 2).view(2, 5, 32)
except RuntimeError as err:
    print("  merge without contiguous():", str(err).split(".")[0])
try:
    AttentionConfig("gqa", 60, 8)
except ValueError as err:
    print("  60 dims into 8 heads:", err)
try:
    AttentionConfig("gqa", 64, 8, num_kv_heads=3)
except ValueError as err:
    print("  8 query heads over 3 KV heads:", err)
q, k = torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 16)
try:
    scaled_dot_product_attention(q, k, k)
except RuntimeError as err:
    print("  q width 8, k width 16:", str(err)[:70])
