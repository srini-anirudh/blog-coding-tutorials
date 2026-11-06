"""FlashAttention on a real GPU: the same output as the unfused math path, with less memory and time.
Section "FlashAttention is not a new attention rule". CUDA only, about 1 minute.

For each length: PyTorch SDPA with the math backend (materialises the scores) against the flash backend,
bf16, causal. Reports median time of 10 calls after warm-up, peak extra memory, and the max difference
of each against an fp32 math reference.
"""

import argparse
import statistics

import _setup  # noqa: F401
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from attnmoves.common import Report

parser = argparse.ArgumentParser()
parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
args = parser.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("needs a CUDA GPU")

B, H, D = 4, 16, 128
rep = Report("09_flash_gpu", device="cuda")
rep(f"B = {B}, H = {H}, d = {D}, causal, bf16. Median of 10 calls after 3 warm-up calls.")


def run(backend, q, k, v):
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


rows = []
for n in args.lengths:
    g = torch.Generator(device="cuda").manual_seed(n)
    q, k, v = (torch.randn(B, H, n, D, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3))
    with torch.no_grad():  # fp32 math reference, one (batch, head) slice at a time to bound its memory
        ref = torch.empty(B, H, n, D, device="cuda")
        for b in range(B):
            for h in range(H):
                ref[b, h] = run(SDPBackend.MATH, q[b, h].float(), k[b, h].float(), v[b, h].float())
    for name, backend in [("math", SDPBackend.MATH), ("flash", SDPBackend.FLASH_ATTENTION)]:
        out = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        try:
            with torch.no_grad():
                for _ in range(3):
                    out = run(backend, q, k, v)
                times = []
                for _ in range(10):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    out = run(backend, q, k, v)
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
            peak = (torch.cuda.max_memory_allocated() - base) / 2**20
            diff = (out.float() - ref).abs().max().item()
            rows.append([n, name, statistics.median(times), peak, diff])
        except torch.OutOfMemoryError:
            rows.append([n, name, "out of memory", "", ""])
        del out
    del ref
rep.table(["n", "backend", "ms per call", "peak extra MiB", "max |out - fp32 ref|"], rows)
rep("Both backends compute the same function. Their bf16 outputs differ from the fp32 reference by rounding.")
rep.save()
