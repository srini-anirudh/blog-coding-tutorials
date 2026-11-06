"""Post section 14, "Quantization as a ledger edit": decode time tracks weight bytes; prefill does not.

A stack of 4096 -> 14336 weight matrices (an FFN up-projection's shape, 1.9 GB in bf16) is run at one activation
row (decode) and at 4,096 rows (prefill) in several weight formats:

- fp32 and bf16 through cuBLAS;
- bf16 and int8 (per-output-channel scale) through one small Triton GEMV kernel written here, so the only thing
  that differs between the two is the bytes per weight;
- int8 at 4,096 rows, dequantized to bf16 and then multiplied, which is what weight-only formats do in prefill.

Quality is not measured: the weights are random. The post says the same: this is a bytes claim.

    python labs/the-mechanics-of-llm-inference/scripts/06_quantization.py --out .../results/quantization_h100.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferlab.timing import Report, measured_bandwidth, time_ms  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def gemv_kernel(w_ptr, x_ptr, s_ptr, y_ptr, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                HAS_SCALE: tl.constexpr):
    """y[n] = scale[n] * sum_k W[n, k] x[k]. W is (N, K) row-major, int8 or bf16; accumulation in fp32."""
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < K), other=0)
        x = tl.load(x_ptr + rk, mask=rk < K, other=0.0)
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    if HAS_SCALE:
        acc = acc * tl.load(s_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    tl.store(y_ptr + rn, acc.to(tl.bfloat16), mask=rn < N)


def gemv(w: torch.Tensor, x: torch.Tensor, scale: torch.Tensor | None, block=(8, 512)) -> torch.Tensor:
    n, k = w.shape
    y = torch.empty(n, dtype=torch.bfloat16, device=w.device)
    bn, bk = block
    gemv_kernel[(triton.cdiv(n, bn),)](w, x, scale if scale is not None else x, y, n, k, BLOCK_N=bn, BLOCK_K=bk,
                                       HAS_SCALE=scale is not None)
    return y


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8: w ~ q * scale."""
    scale = w.float().abs().amax(dim=1) / 127
    q = torch.round(w.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--ffn", type=int, default=14336)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--prefill-rows", type=int, default=4096)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")
    rep = Report(args.out)
    bw = measured_bandwidth()
    d, f, n_layers = args.d, args.ffn, args.layers
    w16 = [torch.randn(f, d, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(n_layers)]
    q8 = [quantize_int8(w) for w in w16]
    x1 = torch.randn(d, device="cuda", dtype=torch.bfloat16)
    rep(f"{n_layers} matrices of {f} x {d}. Measured copy bandwidth {bw / 1e12:.2f} TB/s. Times are per matrix.")

    # The Triton kernel has two block sizes. Pick the fastest for each format, so neither is handicapped.
    blocks = [(4, 1024), (8, 512), (16, 256), (16, 512), (32, 128)]

    def best(fn_of_block):
        return min((time_ms(lambda: fn_of_block(b), warmup=3, iters=10), b) for b in blocks)

    err = (gemv(q8[0][0], x1, q8[0][1]).float() - F.linear(x1, q8[0][0].to(torch.bfloat16) * q8[0][1][:, None]).float())
    rep(f"int8 GEMV against a dequantized reference: max abs diff {err.abs().max().item():.2e}")

    rows = []
    w32 = [w.float() for w in w16[: n_layers // 2]]  # half the stack: fp32 would not leave room otherwise
    t = time_ms(lambda: [F.linear(x1.float(), w) for w in w32], 3, 20) / len(w32)
    rows.append(("decode, 1 row", "fp32 cuBLAS", 4, t))
    del w32
    t = time_ms(lambda: [F.linear(x1, w) for w in w16], 3, 20) / n_layers
    rows.append(("decode, 1 row", "bf16 cuBLAS", 2, t))
    t, b16 = best(lambda b: [gemv(w, x1, None, b) for w in w16])
    rows.append(("decode, 1 row", f"bf16 Triton GEMV {b16}", 2, t / n_layers))
    t, b8 = best(lambda b: [gemv(q, x1, s, b) for q, s in q8])
    rows.append(("decode, 1 row", f"int8 Triton GEMV {b8}", 1, t / n_layers))

    xp = torch.randn(args.prefill_rows, d, device="cuda", dtype=torch.bfloat16)
    t = time_ms(lambda: [F.linear(xp, w) for w in w16], 2, 10) / n_layers
    rows.append((f"prefill, {args.prefill_rows} rows", "bf16 cuBLAS", 2, t))
    t = time_ms(lambda: [F.linear(xp, q.to(torch.bfloat16) * s[:, None]) for q, s in q8], 2, 10) / n_layers
    rows.append((f"prefill, {args.prefill_rows} rows", "int8 dequant, then bf16 cuBLAS", 1, t))

    rep(f"\n{'regime':<20} {'format':<32} {'B/weight':>8} {'us':>8} {'weight GB/s':>11} {'vs bf16 cuBLAS':>14}")
    ref = {r[0]: r[3] for r in rows if r[1] == "bf16 cuBLAS"}
    for regime, fmt, bpw, t in rows:
        gbs = f * d * bpw / (t / 1e3) / 1e9
        rep(f"{regime:<20} {fmt:<32} {bpw:>8} {t * 1e3:>8.1f} {gbs:>11.0f} {ref[regime] / t:>13.2f}x")
    rep("\nweight GB/s counts weight bytes only. In prefill it is not a bandwidth figure, just bytes / time.")
    rep.save()


if __name__ == "__main__":
    main()
