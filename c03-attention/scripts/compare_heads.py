"""MHA, MQA and GQA side by side: parameters, shapes, and proof that one module covers all three."""

import time

import _setup  # noqa: F401
import torch

from attnzoo.config import AttentionConfig
from attnzoo.device import hardware_note
from attnzoo.gqa import GroupedQueryAttention
from attnzoo.heads import repeat_kv, split_heads
from attnzoo.mha import FusedQKVAttention, MultiHeadAttention, MultiQueryAttention


def params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


torch.manual_seed(0)
x = torch.randn(2, 16, 512)

print(f"{'layer':<6}{'Hq':>4}{'Hkv':>5}{'n_rep':>7}{'q':>16}{'k, v':>16}{'params':>10}")
for name, hkv in [("MHA", 32), ("GQA", 8), ("GQA", 2), ("MQA", 1)]:
    cfg = AttentionConfig("gqa", 512, 32, num_kv_heads=hkv)
    m = GroupedQueryAttention(cfg)
    q = split_heads(m.q_proj(x), cfg.num_heads)
    k = split_heads(m.k_proj(x), cfg.num_kv_heads)
    print(f"{name:<6}{cfg.num_heads:>4}{hkv:>5}{cfg.n_rep:>7}{str(tuple(q.shape)):>16}{str(tuple(k.shape)):>16}{params(m):>10,}")

print("\nrepeat_kv: which KV head does each query head read? (Hq = 8, Hkv = 2)")
kv = torch.arange(2.0).view(1, 2, 1, 1)
print(repeat_kv(kv, 4).flatten().int().tolist())

print("\nThe unified module against the separate classes, weights copied:")
mha, mqa = MultiHeadAttention(512, 32, causal=True), MultiQueryAttention(512, 32, causal=True)
for label, src, hkv in [("GQA(Hkv=32) vs MHA", mha, 32), ("GQA(Hkv=1)  vs MQA", mqa, 1)]:
    gqa = GroupedQueryAttention(AttentionConfig("gqa", 512, 32, num_kv_heads=hkv, positional_encoding="none"))
    gqa.load_state_dict(src.state_dict())
    print(f"{label}: max |diff| = {(gqa(x) - src(x)).abs().max().item():.2e}")

print(f"\nThree Linear layers against one fused Linear ({hardware_note()}, projections only):")
big = torch.randn(8, 1024, 512)
fused = FusedQKVAttention.from_separate(mha)
for label, fn in [("separate", lambda: (mha.q_proj(big), mha.k_proj(big), mha.v_proj(big))), ("fused", lambda: fused.qkv_proj(big).chunk(3, -1))]:
    start = time.perf_counter()
    for _ in range(50):
        fn()
    print(f"  {label:<9}{(time.perf_counter() - start) / 50 * 1e3:6.2f} ms per call")
