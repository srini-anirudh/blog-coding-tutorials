"""Inference diagnosis: decode is a bandwidth problem, so batching is nearly free until it isn't.

A synthetic decoder (8 layers, width 4096, 32 heads, bf16, random weights) runs one decode step: each of b
sequences adds one token and attends over a KV cache of L tokens. We time the step for growing b at two
context lengths and compare it with a bytes model: the step must read every weight once plus b * L
cached keys and values per layer. Then a 4096-token prefill through the same layers, for the other regime.

    python scripts/07_decode_batching.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whyslow.bench import Report, cuda_time, need_cuda  # noqa: E402
from whyslow.roofline import H100_SXM, verdict  # noqa: E402

D, H, LAYERS, BF16 = 4096, 32, 8, 2
HD = D // H


def make_layers():
    g = torch.Generator(device="cuda").manual_seed(0)

    def w(*s):
        return torch.randn(*s, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02

    return [{"qkv": w(D, 3 * D), "o": w(D, D), "up": w(D, 4 * D), "down": w(4 * D, D)} for _ in range(LAYERS)]


def forward(layers, x, caches):
    """x: (b, t, D). caches: per layer (k, v), each (b, H, L, HD); new keys are not appended (fixed L)."""
    b, t, _ = x.shape
    for p, (k_cache, v_cache) in zip(layers, caches):
        q, k, v = (x @ p["qkv"]).split(D, dim=-1)
        q = q.view(b, t, H, HD).transpose(1, 2)
        if k_cache is None:  # prefill: attend within the prompt
            k, v = k.view(b, t, H, HD).transpose(1, 2), v.view(b, t, H, HD).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k_cache, v_cache)
        x = x + a.transpose(1, 2).reshape(b, t, D) @ p["o"]
        x = x + F.gelu(x @ p["up"]) @ p["down"]
    return x


def main():
    r = Report("07_decode_batching")
    if not need_cuda(r):
        return r.save()
    bw, peak = H100_SXM["hbm_bytes_per_s"], H100_SXM["bf16_flops"]
    layers = make_layers()
    weight_bytes = sum(t.numel() for p in layers for t in p.values()) * BF16
    weight_flops_per_token = 2 * weight_bytes / BF16
    r(f"weights {weight_bytes / 2**30:.2f} GiB; reading them once at {bw / 1e12:.2f} TB/s takes "
      f"{weight_bytes / bw * 1e3:.2f} ms (vendor bandwidth)\n")
    r(f"{'context':>7s} {'batch':>5s} {'step ms':>8s} {'tokens/s':>9s} {'KV / weights':>12s} {'bytes model ms':>14s} "
      f"{'achieved GB/s':>13s}")
    rows = []
    with torch.no_grad():
        for L in (512, 4096):
            for b in (1, 4, 16, 32, 64, 128):
                kv_bytes = LAYERS * 2 * b * L * D * BF16
                if kv_bytes > 40 * 2**30:
                    continue
                caches = [(torch.randn(b, H, L, HD, device="cuda", dtype=torch.bfloat16),
                           torch.randn(b, H, L, HD, device="cuda", dtype=torch.bfloat16)) for _ in range(LAYERS)]
                x = torch.randn(b, 1, D, device="cuda", dtype=torch.bfloat16)
                t = cuda_time(lambda: forward(layers, x, caches), iters=20)
                model = (weight_bytes + kv_bytes) / bw
                row = {"context": L, "batch": b, "ms": t * 1e3, "tok_s": b / t, "kv_over_w": kv_bytes / weight_bytes,
                       "model_ms": model * 1e3, "gbps": (weight_bytes + kv_bytes) / t / 1e9}
                rows.append(row)
                r(f"{L:7d} {b:5d} {row['ms']:8.3f} {row['tok_s']:9.0f} {row['kv_over_w']:12.2f} {row['model_ms']:14.3f} "
                  f"{row['gbps']:13.0f}")
                del caches
                torch.cuda.empty_cache()

        T = 4096
        x = torch.randn(1, T, D, device="cuda", dtype=torch.bfloat16)
        t = cuda_time(lambda: forward(layers, x, [(None, None)] * LAYERS), iters=10)
        flops = T * weight_flops_per_token + LAYERS * 2 * 2 * T * T * D / 2  # causal attention: half of QK^T and AV
        act_bytes = LAYERS * T * D * BF16 * 12  # rough: activations in and out of each matmul
        v = verdict(flops, weight_bytes + act_bytes, t, peak, bw)
        r(f"\nprefill of {T} tokens, batch 1: {t * 1e3:.2f} ms, {v.row()}")
        b1 = next(x for x in rows if x["batch"] == 1 and x["context"] == 512)
        dv = verdict(weight_flops_per_token, weight_bytes, b1["ms"] / 1e3, peak, bw)
        r(f"decode, batch 1, context 512 (weights only): {dv.row()}")
    r.data = {"decode": rows, "prefill_ms": t * 1e3, "prefill_verdict": vars(v), "decode_verdict": vars(dv)}
    return r.save()


if __name__ == "__main__":
    main()
