"""Recompute the post's analytical numbers from the formulas in rooflab.model. CPU only, under a second.

Every figure here comes from published peaks and shape arithmetic, not from a measurement.
"""

import _setup  # noqa: F401

from rooflab import model as M
from rooflab.hardware import ACCELS, H100, LINKS, TB
from rooflab.report import start

start("01_ridge_points")
s = 2  # bf16

print("\n## One operation, two costs (bf16)")
n = 4096
add_i = M.elementwise_intensity(1, 3, s)
mm_i = M.gemm_intensity(n, n, n, s)
print(f"vector add      I = {add_i:.3f} FLOP/B (any n)")
print(f"matmul n={n}  I = {mm_i:.0f} FLOP/B   ratio {mm_i / add_i:,.0f}x")

print("\n## Table 1: ridge points I* = pi / beta at HBM (FLOP/B)")
print(f"{'accelerator':16} {'beta TB/s':>9} {'bf16':>6} {'fp8':>6} {'fp32':>6}")
for a in ACCELS.values():
    fp8 = f"{a.ridge('fp8'):.0f}" if a.fp8 else "-"
    print(f"{a.name:16} {a.hbm / TB:9.2f} {a.ridge('bf16'):6.0f} {fp8:>6} {a.ridge('fp32'):6.1f}")

ridge = H100.ridge("bf16")
print(f"\n## Overlap: at the ridge, the serial model reaches "
      f"{M.attainable_serial(ridge, H100.bf16, H100.hbm) / H100.bf16:.2f} of the overlapped roof")
print(f"RMSNorm at I = 2 can reach {M.fraction_of_peak(2, ridge):.2%} of H100 bf16 peak")

print("\n## GEMM")
print(f"square bf16 GEMM reaches the H100 ridge at n = {3 * s * ridge / 2:.0f}")
print(f"4096^3 compulsory I = {mm_i:.0f}, {mm_i / ridge:.1f}x past the ridge")
for t in (128, 256):
    print(f"tile {t}x{t}: tiled I -> {M.gemm_intensity_tiled_limit(s, t, t):.0f} FLOP/B")
print(f"tile needed to reach the ridge: T = {s * ridge:.0f}")

print("\n## Table 2: a d x d projection, d = 8192, bf16")
d = 8192
for tokens in (32768, 512, 64, 1):
    i = M.projection_intensity(tokens, d, s)
    side = f"{i / ridge:.1f}x over" if i > ridge else f"{ridge / i:.1f}x under"
    print(f"M = {tokens:6d}  I = {i:7.1f}  {side}")

print("\n## Decode economics, 70B parameters, bf16, H100")
t = M.decode_step_time(70e9, s, H100.hbm)
print(f"batch-1 step = {t * 1e3:.1f} ms, {1 / t:.0f} tokens/s")
print(f"B* = s I* / 2 = {M.batch_to_ridge(ridge, s):.0f}")
kv_tok = M.kv_bytes_per_token(s, 128, 8, 80)
kv_seq = kv_tok * 8192
print(f"KV cache: {kv_tok / 1024:.0f} KiB per token, {kv_seq / 1e9:.2f} GB per 8k sequence")
print(f"batch where KV traffic passes weight traffic: {M.kv_crossover_batch(70e9, s, kv_seq):.0f}")

print("\n## Table 3: elementwise floor, bf16 storage")
ops = [("bf16 -> fp32 cast", 0, 2, 6), ("residual add", 1, 3, 6), ("SwiGLU (elementwise)", 6, 3, 6),
       ("RMSNorm", 5, 2, 4), ("softmax (online)", 6, 2, 4), ("LayerNorm", 9, 2, 4), ("GELU (tanh)", 12, 2, 4)]
for name, f, rw, bpe in ops:
    i = f / bpe
    print(f"{name:22} f={f:2d}  r+w={rw}  {bpe} B/elt  I={i:.2f}  {M.fraction_of_peak(i, ridge):.2%} of peak")
n_act = 8 * 8192 * 8192
print(f"RMSNorm on (8, 8192, 8192) bf16: {M.elementwise_time(n_act, 2, s, H100.hbm) * 1e3:.2f} ms at 3.35 TB/s")

print("\n## Attention, one head, bf16")
print(f"naive, N -> inf: I -> d/s = {128 / s:.0f} (N=4096, d=128: "
      f"{M.attention_flops(4096, 128) / M.attention_bytes_naive(4096, 128, s):.1f})")
print(f"compulsory at N = 2048: I = N/s = "
      f"{M.attention_flops(2048, 128) / M.attention_bytes_compulsory(2048, 128, s):.0f}")
print(f"FA2 tiled, Br = 128, N = 65536: I = "
      f"{M.attention_flops(65536, 128) / M.attention_bytes_flash_tiled(65536, 128, s, 128):.0f} (limit 2Br/s = 128)")
print(f"decode: I = {M.decode_attention_intensity(s):.0f}; GQA g=8: I = {M.decode_attention_intensity(s, 8):.0f}")

print("\n## Optimizers, fp32 master weights, bytes per parameter")
for name in ("sgd", "sgd_momentum", "adam"):
    b = M.optimizer_bytes(name, bf16_copy=True)
    print(f"{name:13} {b.total:3d} B (with bf16 weight copy)  {M.optimizer_bytes(name).total:3d} B (without)")
adam = M.ADAM_POST
print(f"Adam I = {adam.flops / adam.total:.2f}; 7B step = {adam.total * 7e9 / H100.hbm * 1e3:.0f} ms at 3.35 TB/s")

print("\n## Little's law, H100, 600 ns HBM latency")
inflight = M.little_bytes_in_flight(H100.hbm, 600e-9)
print(f"bytes in flight = {inflight / 1e6:.2f} MB, {inflight / 132 / 1024:.1f} KiB per SM, "
      f"{inflight / 132 / 128:.0f} 128-byte requests per SM")

print("\n## The next wall: ridge per link, bf16")
for name, beta in LINKS.items():
    print(f"{name:16} I* = {H100.bf16 / beta:,.0f}")
