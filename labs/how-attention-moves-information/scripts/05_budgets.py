"""The cost arithmetic of the post: KV cache, decode intensity, tensor-parallel sharding, MLA, ViT tokens.
Sections "MHA, MQA, GQA", "Multi-head latent attention" and "Attention over images".

Every number here is computed from a formula and a published model shape. Nothing is measured.
C3 checks the KV formula against real cache tensors: c03-attention/results/bench_kv_memory.txt.
"""

import _setup  # noqa: F401

from attnmoves.common import Report
from attnmoves.budgets import DEEPSEEK_V2, GIB, LLAMA2_7B, decode_intensity, kv_cache_bytes, tp_replication, vit_tokens
from attnmoves.latent import cache_per_token

rep = Report("05_budgets")
m = LLAMA2_7B
rep(f"KV cache for a Llama-2-7B shape (L = {m['layers']}, H = {m['heads']}, dh = {m['dh']}), fp16, S = 8192")
rows = []
for name, hkv in [("MHA", 32), ("GQA Hkv = 8", 8), ("GQA Hkv = 4", 4), ("MQA", 1)]:
    one = kv_cache_bytes(m["layers"], 1, 8192, hkv, m["dh"])
    rows.append([name, hkv, f"{one / GIB:.3f}", f"{64 * one / GIB:.1f}", f"{decode_intensity(8192, m['heads'], hkv, m['dh']):.0f}"])
rep.table(["layer", "Hkv", "GiB, 1 sequence", "GiB, 64 sequences", "decode FLOPs per cache byte"], rows)
rep(f"Weights in fp16: {m['params'] * 2 / GIB:.1f} GiB. At 64 sequences the MHA cache is "
    f"{64 * kv_cache_bytes(m['layers'], 1, 8192, 32, m['dh']) / (m['params'] * 2):.0f}x the weights.")
rep("Decode intensity is Hq / Hkv FLOPs per byte in fp16: grouping is the only lever on it here.")

rep("\nTensor parallelism: copies of each KV head when Hkv heads are split over TP ranks")
rep.table(["Hkv", "TP = 4", "TP = 8", "TP = 16"], [[h, *[tp_replication(h, tp) for tp in (4, 8, 16)]] for h in (1, 4, 8, 32)])

d = DEEPSEEK_V2
mha = cache_per_token(d["heads"], d["dh"])
mla = cache_per_token(d["heads"], d["dh"], d_c=d["d_c"], d_r=d["d_r"])
gqa8 = cache_per_token(d["heads"], d["dh"], kv_heads=8)
rep(f"\nDeepSeek-V2 attention shape (H = {d['heads']}, dh = {d['dh']}, d_c = {d['d_c']}, d_r = {d['d_r']}), numbers per token per layer")
rep.table(["layer", "cached numbers", "fraction of MHA"], [
    ["MHA", mha, 1.0], ["GQA Hkv = 8", gqa8, gqa8 / mha], ["MLA", mla, mla / mha],
])
rep(f"MLA against MHA of the same shape: {100 * (1 - mla / mha):.1f}% smaller. The paper's 93.3% is against its own")
rep("dense 67B model, a different baseline. That figure is quoted, not reproduced.")

rep("\nViT patch tokens with 14-pixel patches")
rows = []
for size in (224, 336, 448, 1024):
    t = vit_tokens(size, 14)
    rows.append([size, t, f"{t * t / 1e6:.1f}"])
rep.table(["image side", "tokens", "score entries per head per layer (millions)"], rows)
rep.save()
