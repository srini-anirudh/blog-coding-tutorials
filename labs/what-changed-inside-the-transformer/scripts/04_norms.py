"""Fewer statistics: LayerNorm vs RMSNorm. And QK-norm: a cap on attention logits.

Post sections: "Axis one: LayerNorm -> RMSNorm" and "QK-norm: the stabilizer nobody expected to need".
The post's throughput argument is about GPUs and tensor parallelism. The CPU timing here is only a sanity check;
the GPU timing is in benchmarks/gpu_bench.py.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from organs.norms import (attention_entropy, layer_norm_two_reductions, qk_logits, rms_norm_one_reduction,
                          softmax_grad_norm)
from organs.report import Report


def median_ms(fn, x, reps: int = 30) -> float:
    fn(x)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t0)
    return sorted(times)[reps // 2] * 1e3


def main() -> None:
    rep = Report("04_norms")
    torch.manual_seed(0)
    x = torch.randn(4096, 1024) * 3 + 0.7  # a stream with a non-zero mean

    rep("1. What each norm computes (no gain, no shift), x of shape (4096, 1024) with mean 0.7:")
    ln, rms = layer_norm_two_reductions(x), rms_norm_one_reduction(x)
    rep(f"   ours vs F.layer_norm, max |diff|                 {(ln - F.layer_norm(x, (1024,))).abs().max():.2e}")
    rep(f"   ours vs F.rms_norm, max |diff|                   {(rms - F.rms_norm(x, (1024,), eps=1e-5)).abs().max():.2e}")
    rep(f"   LN(x) vs RMS(x - mean(x)), max |diff|            "
        f"{(ln - rms_norm_one_reduction(x - x.mean(-1, keepdim=True))).abs().max():.2e}")
    rep(f"   LN(x) vs RMS(x) on this x, max |diff|            {(ln - rms).abs().max():.2e}   (they differ by the mean)")
    xc = x - x.mean(-1, keepdim=True)
    rep(f"   LN vs RMS once the mean is zero, max |diff|      "
        f"{(layer_norm_two_reductions(xc) - rms_norm_one_reduction(xc)).abs().max():.2e}")
    rep("   LayerNorm = center, then RMSNorm. RMSNorm drops the centering and its reduction.")
    rep("")
    rep("   CPU timing, median of 30, (16384, 4096) fp32. A sanity check, not the post's GPU argument:")
    big = torch.randn(16384, 4096)
    rows = [["two reductions (ours)", f"{median_ms(layer_norm_two_reductions, big):.1f}"],
            ["one reduction (ours)", f"{median_ms(rms_norm_one_reduction, big):.1f}"],
            ["F.layer_norm", f"{median_ms(lambda t: F.layer_norm(t, (4096,)), big):.1f}"],
            ["F.rms_norm", f"{median_ms(lambda t: F.rms_norm(t, (4096,)), big):.1f}"]]
    rep.table(["   op", "ms"], rows)
    rep("")

    rep("2. QK-norm. One head, d = 256, d_h = 64, T = 128, causal. Scale W_q and W_k by s, as if training had")
    rep("   grown them. Without QK-norm the logits grow as s^2 and softmax saturates; with it they stay bounded.")
    g = torch.Generator().manual_seed(0)
    xs = torch.randn(4, 128, 256, generator=g, dtype=torch.float64)
    wq0 = torch.randn(256, 64, generator=g, dtype=torch.float64) / 16
    wk0 = torch.randn(256, 64, generator=g, dtype=torch.float64) / 16
    rows = []
    for s in (1, 2, 4, 8, 16):
        cells = [s]
        for qk in (False, True):
            z = qk_logits(xs, wq0 * s, wk0 * s, qk)
            p = z.softmax(-1)
            cells += [f"{z[torch.isfinite(z)].abs().max():.1f}", f"{attention_entropy(p[:, 64:]):.2f}",
                      f"{softmax_grad_norm(z):.3f}"]
        rows.append(cells)
    rep.table(["s", "max|logit|", "entropy", "softmax grad", "max|logit| QK", "entropy QK", "softmax grad QK"], rows)
    rep("   Entropy is over query rows 64..127 (uniform = log 65..128, about 4.5 nats).")
    rep("   bf16 has 8 bits of mantissa: at a logit of 100 its spacing is 0.5, so large logits also lose precision.")
    rep(f"   bf16 spacing at 100: {(torch.tensor(100.0, dtype=torch.bfloat16).nextafter(torch.tensor(200.0, dtype=torch.bfloat16)) - 100).item()}")
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    main()
