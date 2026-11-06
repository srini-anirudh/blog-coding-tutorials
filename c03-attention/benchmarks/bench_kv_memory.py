"""Verify M_KV = 2 L B S Hkv dh b against the tensors a real forward pass leaves in the cache.

Run:  python c03-attention/benchmarks/bench_kv_memory.py
Part 1 measures actual cache tensors (and the MPS allocator). Part 2 applies the
verified formula to full-size configurations; those numbers are computed, not measured.
"""

from __future__ import annotations

import _common  # noqa: F401  (sets sys.path)
import torch

from attnzoo.config import AttentionConfig
from attnzoo.kv_cache import KVCache, kv_cache_bytes, mla_cache_bytes
from attnzoo.library import build_attention


def layers(kind: str, dtype: torch.dtype, device: str, num_layers: int, max_seq_len: int) -> tuple[list, AttentionConfig]:
    d, h = 1024, 16
    # A tiny causal window keeps attention cheap: the cache contents do not depend on it.
    common = dict(hidden_size=d, num_heads=h, max_seq_len=max_seq_len, window_size=64, backend="window_chunked")
    if kind == "MLA":
        common["backend"] = "sdpa"
        common["window_size"] = 32
        cfg = AttentionConfig("mla", kv_lora_rank=128, qk_rope_head_dim=16, **common)
    else:
        kv = {"MHA": 16, "GQA": 4, "MQA": 1}[kind]
        cfg = AttentionConfig("gqa", num_kv_heads=kv, **common)
    return [build_attention(cfg, layer_idx=i).to(device=device, dtype=dtype).eval() for i in range(num_layers)], cfg


def formula(kind: str, cfg: AttentionConfig, num_layers: int, b: int, s: int, bytes_per_elem: int) -> int:
    if kind == "MLA":
        return mla_cache_bytes(num_layers, b, s, cfg.kv_lora_rank, cfg.qk_rope_head_dim, bytes_per_elem)
    return kv_cache_bytes(num_layers, b, s, cfg.num_kv_heads, cfg.head_dim, bytes_per_elem)


@torch.inference_mode()
def measure(kind: str, s: int, dtype: torch.dtype, device: str) -> tuple[int, int, int | None]:
    num_layers, b = 2, 1
    ls, cfg = layers(kind, dtype, device, num_layers, s + 8)
    cache = KVCache()
    x = torch.randn(b, s, cfg.hidden_size, device=device, dtype=dtype)
    base = torch.mps.current_allocated_memory() if device == "mps" else None
    for layer in ls:
        layer(x, kv_cache=cache)
    alloc = None
    if device == "mps":
        torch.mps.synchronize()
        alloc = torch.mps.current_allocated_memory() - base
    return formula(kind, cfg, num_layers, b, s, torch.empty(0, dtype=dtype).element_size()), cache.nbytes(), alloc


def main() -> None:
    torch.set_num_threads(2)
    print("Part 1: measured. L = 2, B = 1, D = 1024, Hq = 16, dh = 64; GQA Hkv = 4; MLA r = 128, dr = 16.")
    print(f"{'layer':6}{'S':>7}{'dtype':>10}{'device':>8}{'formula':>14}{'tensors':>14}{'equal':>7}{'MPS alloc delta':>18}")
    for device in ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]:
        for dtype, name in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
            for kind in ["MHA", "GQA", "MQA", "MLA"]:
                lengths = [1024, 16384] if device == "cpu" and dtype == torch.float32 else [1024]
                for s in lengths if kind != "MLA" else [1024]:  # MLA has no chunked path: keep it short
                    f, m, alloc = measure(kind, s, dtype, device)
                    extra = f"{alloc:,}" if alloc is not None else "-"
                    print(f"{kind:6}{s:>7}{name:>10}{device:>8}{f:>14,}{m:>14,}{str(f == m):>7}{extra:>18}")

    print("\nPart 2: computed from the verified formula, not measured. L = 32, D = 4096, Hq = 32, dh = 128, B = 1, bf16 (b = 2).")
    print(f"{'layer':16}{'numbers per token':>19}{'S = 8K':>11}{'S = 128K':>11}{'S = 1M':>11}")
    rows = [("MHA Hkv=32", 2 * 32 * 128), ("GQA Hkv=8", 2 * 8 * 128), ("GQA Hkv=2", 2 * 2 * 128), ("MQA Hkv=1", 2 * 128), ("MLA r=512 dr=64", 512 + 64)]
    for name, per_token in rows:
        sizes = [32 * s * per_token * 2 / 2**30 for s in (8192, 131072, 1048576)]
        print(f"{name:16}{per_token:>19}" + "".join(f"{x:>8.1f} GiB" for x in sizes))


if __name__ == "__main__":
    main()
