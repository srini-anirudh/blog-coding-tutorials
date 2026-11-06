"""Build MLA in the four steps of the post, printing what each one creates."""

import _setup  # noqa: F401
import torch

from attnzoo.config import AttentionConfig
from attnzoo.gqa import GroupedQueryAttention
from attnzoo.kv_cache import KVCache
from attnzoo.mla import MultiHeadLatentAttention

torch.manual_seed(0)
cfg = AttentionConfig(
    "mla", hidden_size=256, num_heads=8, head_dim=32, kv_lora_rank=32, qk_rope_head_dim=8, v_head_dim=32, max_seq_len=512
)
mla = MultiHeadLatentAttention(cfg).eval()
x = torch.randn(1, 16, 256)


def show(name: str, t: torch.Tensor) -> None:
    print(f"  {name:<8} {tuple(t.shape)}")


print("Step 1, compress: this is all we store")
c_kv, k_pe = mla.compress(x)
show("c_kv", c_kv)
show("k_pe", k_pe)
print("Step 2, reconstruct per-head keys and values on the fly")
k_nope, v = mla.reconstruct(c_kv)
show("k_nope", k_nope)
show("v", v)
print("Step 3, RoPE lives only in the small shared key")
q_nope, q_pe = mla.queries(x)
show("q_nope", q_nope)
show("q_pe", q_pe)
print(f"  full query and key width: {q_nope.size(-1)} + {q_pe.size(-1)} = {q_nope.size(-1) + q_pe.size(-1)}")

print("\nDecoding, three ways:")
full = mla(x)
for name, absorb in [("explicit K, V", False), ("absorbed", True)]:
    cache = KVCache()
    steps = [mla(x[:, :10], kv_cache=cache, absorb=absorb)]
    steps += [mla(x[:, t : t + 1], kv_cache=cache, absorb=absorb) for t in range(10, 16)]
    err = (torch.cat(steps, 1) - full).abs().max().item()
    print(f"  {name:<14} max |cached - full| = {err:.1e}, cached tensors {tuple(cache.keys[0].shape)} {tuple(cache.values[0].shape)}")

print("\nStep 4, cache bytes per token for one layer (fp32):")
x = torch.randn(1, 100, 256)
rows = [
    ("MHA", AttentionConfig("mha", 256, 8), GroupedQueryAttention),
    ("GQA Hkv=2", AttentionConfig("gqa", 256, 8, num_kv_heads=2), GroupedQueryAttention),
    ("MQA", AttentionConfig("mqa", 256, 8), GroupedQueryAttention),
    ("MLA r=32", cfg, MultiHeadLatentAttention),
]
base = None
for name, c, cls in rows:
    cache = KVCache()
    cls(c)(x, kv_cache=cache)
    per_token = cache.nbytes() / 100
    base = base or per_token
    print(f"  {name:<11}{per_token:>7.0f} bytes   {base / per_token:5.1f}x smaller than MHA")
print("\nAn honest catch: MLA spends cache savings on parameters and compute.")
for name, c, cls in rows:
    print(f"  {name:<11}{sum(p.numel() for p in cls(c).parameters()):>9,} params")
