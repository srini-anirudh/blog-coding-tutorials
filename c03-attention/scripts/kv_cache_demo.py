"""Prefill and decode with a KV cache: shapes, offsets, and cached == full."""

import _setup  # noqa: F401
import torch

from attnzoo.config import AttentionConfig
from attnzoo.gqa import GroupedQueryAttention
from attnzoo.kv_cache import KVCache, StaticKVCache, kv_cache_bytes
from attnzoo.rope import RotaryEmbedding

torch.manual_seed(0)
cfg = AttentionConfig("gqa", hidden_size=64, num_heads=8, num_kv_heads=2, max_seq_len=64)
attn = GroupedQueryAttention(cfg).eval()
x = torch.randn(1, 12, 64)
full = attn(x)

print("RoPE tables:", tuple(attn.rotary.cos_cached.shape), "| first pair of dims at positions 0..3:")
print(attn.rotary.cos_cached[:4, 0], attn.rotary.sin_cached[:4, 0])
print("rotation speeds:", attn.rotary.inv_freq)

print("\nstep        Nq  past  Nk   k in cache        bytes  max |cached - full|")
cache = KVCache()
outs = []
start = 0
for n in [8, 1, 1, 1, 1]:
    chunk = x[:, start : start + n]
    out = attn(chunk, kv_cache=cache)
    outs.append(out)
    name = "prefill" if start == 0 else "decode"
    err = (out - full[:, start : start + n]).abs().max().item()
    print(f"{name:<11}{n:>3}{start:>6}{start + n:>4}   {str(tuple(cache.keys[0].shape)):<18}{cache.nbytes():>10,}    {err:.1e}")
    start += n

expected = kv_cache_bytes(1, 1, 12, 2, 8, 4)
print(f"\nformula 2 L B S Hkv dh b = 2*1*1*12*2*8*4 = {expected:,} bytes; measured {cache.nbytes():,} bytes")

print("\nPreallocated: the buffer never moves.")
static = StaticKVCache(num_layers=1, batch_size=1, num_kv_heads=2, max_seq_len=32, head_dim=8)
address = static.k.data_ptr()
static_outs = [attn(x[:, :8], kv_cache=static)] + [attn(x[:, t : t + 1], kv_cache=static) for t in range(8, 12)]
print("buffer:", tuple(static.k.shape), "| same address:", static.k.data_ptr() == address, "| filled:", static.get_seq_length())
print(f"max |static - full| = {(torch.cat(static_outs, 1) - full).abs().max().item():.1e}")
print("reserved:", static.nbytes(), "bytes, whatever the sequence length")

print("\nPositions continue from the cache length. What the decode step used:")
decode_pos = torch.arange(cache.get_seq_length() - 1, cache.get_seq_length())[None]
print("position_ids for the last decode step:", decode_pos.tolist())
rope = RotaryEmbedding(8, 64)
print("cos at that position differs from position 0:", (rope(decode_pos)[0] - rope(torch.tensor([[0]]))[0]).abs().max().item() > 0)
