"""GPU measurements for three of the post's systems claims. Needs CUDA; exits with a message otherwise.

1. Decode attention is memory-bound: one decode step over a 32k-token cache, n_q = 32 query heads, n_kv in
   {32, 8, 4, 1}. If the post is right, time tracks the KV bytes read and the achieved bandwidth sits near HBM peak,
   while FLOP/s stays far below the tensor-core peak.
2. LayerNorm vs RMSNorm on (65536, 4096) bf16: PyTorch's kernels and the eager two-reduction / one-reduction code.
3. Peak memory of full vs chunked cross-entropy for S = 8192 tokens, d = 4096, V = 151,936.

    python labs/what-changed-inside-the-transformer/benchmarks/gpu_bench.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from organs.norms import layer_norm_two_reductions, rms_norm_one_reduction
from organs.report import Report
from organs.small import chunked_cross_entropy


def cuda_ms(fn, reps: int = 50) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[reps // 2]


def grouped_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q (B, n_q, 1, d), k, v (B, n_kv, s, d). Each KV head is read once for its whole group of query heads."""
    b, nq, _, d = q.shape
    nkv = k.shape[1]
    qg = q.view(b, nkv, nq // nkv, d)  # (B, n_kv, group, d)
    p = (qg @ k.transpose(-1, -2) / d**0.5).softmax(-1)  # (B, n_kv, group, s)
    return (p @ v).view(b, nq, 1, d)


def main() -> None:
    if not torch.cuda.is_available():
        print("No CUDA device: this benchmark only runs on a GPU.")
        return
    rep = Report("gpu_bench")
    dev = torch.device("cuda")
    rep("1. One decode step of attention, batch 8, n_q = 32, d_h = 128, s = 32,768, bf16. Median of 50.")
    rows = []
    b, nq, d, s = 8, 32, 128, 32768
    q = torch.randn(b, nq, 1, d, device=dev, dtype=torch.bfloat16)
    for nkv in (32, 8, 4, 1):
        k = torch.randn(b, nkv, s, d, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        kv_bytes = 2 * k.numel() * k.element_size()
        flops = 4 * b * nq * d * s
        ms_g = cuda_ms(lambda: grouped_decode(q, k, v))
        ms_sdpa = cuda_ms(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
        rows.append([nkv, f"{kv_bytes / 2**20:.0f}", f"{ms_g:.3f}", f"{kv_bytes / ms_g / 1e6:.0f}",
                     f"{flops / ms_g / 1e9:.1f}", f"{ms_sdpa:.3f}", f"{kv_bytes / ms_sdpa / 1e6:.0f}"])
        del k, v
    rep.table(["n_kv", "KV MiB read", "grouped ms", "grouped GB/s", "grouped TFLOP/s", "SDPA ms", "SDPA GB/s"], rows)
    rep("   H100 SXM spec: 3.35 TB/s HBM, about 989 dense bf16 TFLOP/s. Grouped = einsum-style code that reads")
    rep("   each KV head once per group. SDPA = F.scaled_dot_product_attention(enable_gqa=True), whatever kernel")
    rep("   PyTorch picks for a single query.")
    rep("")

    rep("2. Normalization over (65536, 4096) bf16, forward only. Median of 50.")
    x = torch.randn(65536, 4096, device=dev, dtype=torch.bfloat16)
    nbytes = 2 * x.numel() * x.element_size()  # read once, write once
    rows = []
    for name, fn in [("F.layer_norm", lambda: F.layer_norm(x, (4096,))), ("F.rms_norm", lambda: F.rms_norm(x, (4096,))),
                     ("two reductions, eager", lambda: layer_norm_two_reductions(x)),
                     ("one reduction, eager", lambda: rms_norm_one_reduction(x))]:
        ms = cuda_ms(fn)
        rows.append([name, f"{ms:.3f}", f"{nbytes / ms / 1e6:.0f}"])
    rep.table(["op", "ms", "effective GB/s (1 read + 1 write)"], rows)
    del x
    rep("")

    rep("3. Cross-entropy peak memory, S = 8192 tokens, d = 4096, V = 151,936, bf16 weights, fp32 logits.")
    n, dm, vocab = 8192, 4096, 151936
    h = torch.randn(n, dm, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(vocab, dm, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    y = torch.randint(0, vocab, (n,), device=dev)
    rows = []
    for name, fn in [("full", lambda: F.cross_entropy((h @ w.T).float(), y)),
                     ("chunked 1024", lambda: chunked_cross_entropy(h, w, y, 1024)),
                     ("chunked 256", lambda: chunked_cross_entropy(h, w, y, 256))]:
        h.grad = w.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        loss = fn()
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        rows.append([name, f"{loss.item():.4f}", f"{peak / 2**30:.2f}"])
    rep.table(["cross-entropy", "loss", "peak extra GiB (fwd + bwd)"], rows)
    rep(f"   One fp32 logits tensor is {n * vocab * 4 / 2**30:.2f} GiB. Peak extra includes the weight and hidden grads"
        f" ({(w.numel() + h.numel()) * 2 / 2**30:.2f} GiB).")
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    main()
