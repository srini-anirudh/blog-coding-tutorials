"""MFU: check the FLOP formula with FlopCounterMode, measure a peak, compute utilisation.

    python c05-pretraining/benchmarks/bench_mfu.py --metrics runs/tinystories/metrics.jsonl
    python c05-pretraining/benchmarks/bench_mfu.py --device cuda --peak-tflops 248   # a datasheet peak
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from pretrain.config import TrainConfig  # noqa: E402
from pretrain.metrics import read_metrics  # noqa: E402
from pretrain.mfu import flops_per_token, matmul_params, measure_matmul_peak, mfu  # noqa: E402
from pretrain.model import build_model, count_params  # noqa: E402
from pretrain.step import lm_loss_sum  # noqa: E402
from pretrain.utils import ROOT, pick_device, resolve  # noqa: E402


def counted_flops_per_token(model, a, seq_len: int) -> float:
    """FlopCounterMode reports 2 FLOPs per multiply-add, the same convention as the formula."""
    x = torch.randint(0, a.vocab_size, (1, seq_len))
    with FlopCounterMode(display=False) as fc:
        lm_loss_sum(model(x).logits, torch.roll(x, -1, 1)).backward()
    model.zero_grad(set_to_none=True)
    return fc.get_total_flops() / seq_len


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/tinystories.yaml"))
    ap.add_argument("--metrics", default="runs/tinystories/metrics.jsonl", help="a training run on --device")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32", help="the training run's precision")
    ap.add_argument("--peak-tflops", type=float, help="use this datasheet peak instead of measuring one")
    args = ap.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    a = cfg.model
    model = build_model(a)
    n, fpt = matmul_params(model, a), flops_per_token(model, a, cfg.seq_len)
    print(f"params {count_params(model):,}, matmul params N {n:,}")
    print(f"formula  6N + 12 n_layers d T = {6 * n:,} + {12 * a.num_layers * a.hidden_size * cfg.seq_len:,} "
          f"= {fpt:,.0f} FLOPs per token")
    counted = counted_flops_per_token(model, a, cfg.seq_len)
    print(f"FlopCounterMode (forward + backward): {counted:,.0f} FLOPs per token, "
          f"ratio formula/counted {fpt / counted:.2f}")

    device = pick_device(args.device)
    if args.peak_tflops:
        peak = args.peak_tflops * 1e12
        print(f"\ndatasheet peak given: {args.peak_tflops} TFLOP/s")
    else:
        dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
        peak = measure_matmul_peak(device, dtype, n=2048 if device.type == "cpu" else 4096)
        print(f"\nmeasured peak, best large square matmul: {device} {dtype} {peak / 1e12:.3f} TFLOP/s")

    path = resolve(args.metrics)
    if path.exists():
        rows = [r["tok_s"] for r in read_metrics(path) if "tok_s" in r and r["step"] > 20]
        tok_s = statistics.median(rows)
        print(f"training run: median {tok_s:,.0f} tokens/s on {device} ({args.precision})")
        print(f"achieved {tok_s * fpt / 1e12:.3f} TFLOP/s, MFU against that peak {100 * mfu(tok_s, fpt, peak):.1f}%")


if __name__ == "__main__":
    main()
