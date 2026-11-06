"""The unified library: one config, any architecture on any backend that supports it."""

from pathlib import Path

import _setup  # noqa: F401
import torch
import yaml

from attnzoo import AttentionConfig, build_attention, cache_elems_per_token, make_cache

torch.manual_seed(0)
print(f"{'attention_type':<16}{'class':<28}{'params':>9}{'cache/token':>13}")
for t, extra in [("mha", {}), ("mqa", {}), ("gqa", {"num_kv_heads": 2}), ("mla", {}), ("linear", {})]:
    cfg = AttentionConfig(t, hidden_size=256, num_heads=8, **extra)
    m = build_attention(cfg)
    print(f"{t:<16}{type(m).__name__:<28}{sum(p.numel() for p in m.parameters()):>9,}{cache_elems_per_token(cfg):>13}")

print("\nThe same GQA weights on every backend (max |diff| against naive):")
x = torch.randn(1, 64, 256)
ref_cfg = AttentionConfig("gqa", 256, 8, num_kv_heads=2, backend="naive")
ref = build_attention(ref_cfg).eval()
base = ref(x)
for backend in ["sdpa", "sdpa_math", "sdpa_flash", "tiled"]:
    m = build_attention(AttentionConfig("gqa", 256, 8, num_kv_heads=2, backend=backend)).eval()
    m.load_state_dict(ref.state_dict())
    print(f"  {backend:<12}{(m(x) - base).abs().max().item():.1e}")

print("\nA window can change the backend's cost without changing the answer:")
a = build_attention(AttentionConfig("gqa", 256, 8, num_kv_heads=2, window_size=16, backend="sdpa")).eval()
b = build_attention(AttentionConfig("gqa", 256, 8, num_kv_heads=2, window_size=16, backend="window_chunked")).eval()
b.load_state_dict(a.state_dict())
print(f"  sdpa + mask vs window_chunked: {(a(x) - b(x)).abs().max().item():.1e}")

print("\nConfigs in configs/ (built on the meta device: no memory is allocated):")
for path in sorted((Path(__file__).parent.parent / "configs").glob("*.yaml")):
    cfg = AttentionConfig(**yaml.safe_load(path.read_text()))
    with torch.device("meta"):
        m = build_attention(cfg)
    print(f"  {path.name:<20}{type(m).__name__:<28}{sum(p.numel() for p in m.parameters()):>12,} params, {cache_elems_per_token(cfg):>6} cached numbers per token per layer")

print("\nDecode-time state from make_cache:")
for t in ["gqa", "mla", "linear"]:
    print(f"  {t:<8}{type(make_cache(AttentionConfig(t, 256, 8))).__name__}")
