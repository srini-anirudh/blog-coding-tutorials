"""Attention is a data-movement problem. GPU, bf16, d = 128.

Part 1 (prefill): naive attention that materialises S and P in HBM, against PyTorch SDPA's flash backend. Same W,
different Q. The naive intensity tends to d/s = 64 as N grows; the flash kernel's grows with N.
Part 2 (decode): one query token against a KV cache, with g query heads sharing each KV head (GQA). The bytes are
the cache; the FLOPs grow with g. Time should follow the bytes.

The attention implementations themselves are built step by step in tutorial C3 (c03-attention).
"""

import _setup  # noqa: F401
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from rooflab import model as M
from rooflab.report import measured_ceilings, save_json, start
from rooflab.timing import auto_time

start("08_attention")
c = measured_ceilings()
beta, pi = c["beta"], c["bf16"]
print(f"beta = {beta / 1e12:.2f} TB/s, bf16 pi = {pi / 1e12:.0f} TFLOP/s ({c['source']})")
dev, bf, s, d = "cuda", torch.bfloat16, 2, 128


def naive(q, k, v):
    p = torch.softmax(q @ k.transpose(-1, -2) * d**-0.5, dim=-1)  # S and P are full N x N tensors in HBM
    return p @ v


def flash(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


heads = 16
print(f"\n## Part 1: prefill, batch 1, {heads} heads, non-causal")
print(f"{'N':>6} {'I naive':>8} {'naive ms':>9} {'naive TF/s':>11} {'flash ms':>9} {'flash TF/s':>11} {'speed-up':>9}")
prefill = []
for n in (512, 1024, 2048, 4096, 8192, 16384):
    q, k, v = (torch.randn(1, heads, n, d, device=dev, dtype=bf) for _ in range(3))
    w = heads * M.attention_flops(n, d)
    i_naive = M.attention_flops(n, d) / M.attention_bytes_naive(n, d, s)
    t_n = auto_time(lambda: naive(q, k, v), budget=0.1)
    t_f = auto_time(lambda: flash(q, k, v), budget=0.1)
    prefill.append({"N": n, "I_naive": i_naive, "t_naive": t_n, "t_flash": t_f, "W": w})
    print(f"{n:6d} {i_naive:8.1f} {t_n * 1e3:9.3f} {w / t_n / 1e12:11.1f} {t_f * 1e3:9.3f} "
          f"{w / t_f / 1e12:11.1f} {t_n / t_f:8.1f}x")
    del q, k, v
    torch.cuda.empty_cache()
print(f"Naive roof at I = 64: beta x 64 = {beta * 64 / 1e12:.0f} TFLOP/s. The naive model counts S and P as bf16 "
      "and four N^2 passes; the real kernels add a scale and softmax pass of their own.")

batch, n_ctx, h_kv = 8, 32768, 8
print(f"\n## Part 2: decode, batch {batch}, cache N = {n_ctx}, {h_kv} KV heads, one new token")
kc = torch.randn(batch, h_kv, n_ctx, d, device=dev, dtype=bf)
vc = torch.randn_like(kc)
kv_bytes = 2 * kc.numel() * s
print(f"KV cache read per step: {kv_bytes / 1e9:.2f} GB; model time at measured beta: {kv_bytes / beta * 1e6:.0f} us")
print(f"{'g':>3} {'q heads':>8} {'I':>5} {'time us':>8} {'TB/s':>6} {'TFLOP/s':>8}")
decode = []
for g in (1, 2, 4, 8):
    q = torch.randn(batch, h_kv, g, d, device=dev, dtype=bf)  # g query rows share one K and V

    def step():
        p = torch.softmax(q @ kc.transpose(-1, -2) * d**-0.5, dim=-1)
        return p @ vc

    t = auto_time(step, budget=0.1)
    w = batch * h_kv * g * 4 * n_ctx * d
    decode.append({"g": g, "t": t, "W": w, "bw": kv_bytes / t})
    print(f"{g:3d} {h_kv * g:8d} {M.decode_attention_intensity(s, g):5.0f} {t * 1e6:8.0f} {kv_bytes / t / 1e12:6.2f} "
          f"{w / t / 1e12:8.2f}")
print(f"g = 8 does {decode[-1]['W'] / decode[0]['W']:.0f}x the FLOPs of g = 1 in "
      f"{decode[-1]['t'] / decode[0]['t']:.2f}x the time.")
save_json("08_attention", {"beta": beta, "prefill": prefill, "decode": decode})
