"""fp32 versus bf16 autocast: step time, tokens/s, and how far the losses drift apart.

Both runs start from the same weights and see the same batches, so any loss difference
comes from numerics alone.

    python c05-pretraining/benchmarks/bench_precision.py --device mps --steps 30
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pretrain.config import TrainConfig  # noqa: E402
from pretrain.loader import PackedWindows, ResumableLoader  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.optim import build_optimizer  # noqa: E402
from pretrain.shards import TokenShards  # noqa: E402
from pretrain.step import train_step  # noqa: E402
from pretrain.utils import ROOT, pick_device, resolve, seed_everything, synchronize  # noqa: E402


def run(cfg: TrainConfig, model, device, precision: str, steps: int) -> tuple[list[float], list[float]]:
    model = copy.deepcopy(model).to(device)
    opt = build_optimizer(model, cfg, device)
    loader = ResumableLoader(PackedWindows(TokenShards(resolve(cfg.data_dir), "train"), cfg.seq_len),
                             cfg.micro_batch_size, cfg.seed)
    losses, times = [], []
    for _ in range(steps):
        micro = [next(loader) for _ in range(cfg.grad_accum_steps)]
        synchronize(device)
        t = time.perf_counter()
        res = train_step(model, opt, None, micro, device, precision, cfg.grad_clip)
        synchronize(device)
        times.append(time.perf_counter() - t)
        losses.append(res.loss)
    return losses, times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/tinystories.yaml"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    device = pick_device(args.device)
    seed_everything(0)
    model = build_model(cfg.model, cfg.model_type)
    tokens = cfg.tokens_per_step

    results = {p: run(cfg, model, device, p, args.steps) for p in ("fp32", "bf16")}
    print(f"device {device}, {tokens:,} tokens/step, {args.steps} steps (first 3 dropped as warmup)")
    print(f"{'precision':<10}{'median step ms':>16}{'tokens/s':>12}{'final loss':>12}")
    for p, (losses, times) in results.items():
        med = statistics.median(times[3:])
        print(f"{p:<10}{1000 * med:>16.1f}{tokens / med:>12,.0f}{losses[-1]:>12.4f}")
    diff = [abs(a - b) for a, b in zip(results["fp32"][0], results["bf16"][0])]
    print(f"max |loss_fp32 - loss_bf16| over the run: {max(diff):.4f}")


if __name__ == "__main__":
    main()
