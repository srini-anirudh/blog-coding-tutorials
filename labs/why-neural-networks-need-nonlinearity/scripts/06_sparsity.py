"""Section 06: exact zeros are a systems property. Near-zeros are not.

(a) Fig 08: exact-zero and near-zero fractions after each activation, for Gaussian and heavy-tailed inputs.
(b) Skipping W_down rows. A token's FFN output reads only the rows of W_down^T whose activation is nonzero. The
    result is identical to the dense product, and the bytes read shrink with the sparsity. Timed at the
    Llama-2-7B FFN size (d=4096, h=11008) for one token on CPU, which is the memory-bound decode case.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from nonlin import activations as A  # noqa: E402
from nonlin.ffn import sparse_down  # noqa: E402
from nonlin.stats import Results, exact_zero_fraction, near_zero_fraction  # noqa: E402

ACTS = ["relu", "relu2", "gelu", "gelu_tanh", "silu"]


def part_a(res: Results, n: int) -> None:
    gen = torch.Generator().manual_seed(0)
    t = torch.distributions.StudentT(4.0)
    torch.manual_seed(0)
    inputs = {
        "gaussian": torch.randn(n, generator=gen),
        "student-t(4), unit variance": t.sample((n,)) / (2.0**0.5),  # Var of t(4) is 2
    }
    out = {}
    for label, z in inputs.items():
        res.print(f"(a) {n} float32 pre-activations, {label}: fraction of outputs with |f| <= eps")
        res.print(f"{'activation':<11} {'exactly 0':>10} {'eps=1e-3':>9} {'eps=1e-2':>9} {'eps=0.1':>9}")
        rows = {}
        for act in ACTS:
            y = A.get(act)(z)
            row = {"exact": exact_zero_fraction(y), **{str(e): near_zero_fraction(y, e) for e in (1e-3, 1e-2, 0.1)}}
            res.print(f"{act:<11} {row['exact']:>10.4f} {row['0.001']:>9.4f} {row['0.01']:>9.4f} {row['0.1']:>9.4f}")
            rows[act] = row
        out[label] = rows
    res.data["zeros"] = out


def median_ms(fn, reps: int) -> float:
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def part_b(res: Results, d: int, h: int, reps: int) -> None:
    torch.manual_seed(0)
    w_t = torch.randn(h, d) / d**0.5  # W_down^T, (h, d), float32
    res.print(f"\n(b) One token through W_down: d={d}, h={h}, float32 ({w_t.numel() * 4 / 2**20:.0f} MiB), "
              f"median of {reps} runs, {torch.get_num_threads()} threads")
    res.print(f"{'exact-zero fraction':>20} {'rows read':>10} {'MiB read':>9} {'max |sparse-dense|':>19} "
              f"{'dense ms':>9} {'sparse ms':>10} {'speed-up':>9}")
    out = {}
    for shift in [0.0, 1.2816, 2.3263]:  # Gaussian quantiles for 50%, 90% and 99% zeros
        a = A.relu(torch.randn(h) - shift)
        dense = a @ w_t
        sparse, rows = sparse_down(a, w_t)
        err = (sparse - dense).abs().max().item()
        t_dense = median_ms(lambda: a @ w_t, reps)
        t_sparse = median_ms(lambda: sparse_down(a, w_t), reps)
        frac = exact_zero_fraction(a)
        res.print(f"{frac:>20.3f} {rows:>10} {rows * d * 4 / 2**20:>9.1f} {err:>19.2e} {t_dense:>9.2f} "
                  f"{t_sparse:>10.2f} {t_dense / t_sparse:>9.2f}")
        out[f"{frac:.3f}"] = {"rows_read": rows, "max_abs_err": err, "dense_ms": t_dense, "sparse_ms": t_sparse}
    gelu_a = A.gelu(torch.randn(h))
    _, rows = sparse_down(gelu_a, w_t)
    res.print(f"Same trick on a GELU activation: {rows} of {h} rows read (no exact zeros to skip)")
    out["gelu_rows_read"] = rows
    res.data["skip_rows"] = out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path, default=LAB / "results")
    args = p.parse_args()
    res = Results("06_sparsity", args.out)
    part_a(res, args.n)
    if args.quick:
        part_b(res, 256, 688, 3)
    else:
        part_b(res, 4096, 11008, args.reps)
    res.save()


if __name__ == "__main__":
    main()
