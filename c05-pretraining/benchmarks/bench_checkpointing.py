"""Activation checkpointing: memory held for backward against step time.

    python c05-pretraining/benchmarks/bench_checkpointing.py --device mps
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pretrain.activation_ckpt import apply_activation_checkpointing, saved_tensor_bytes  # noqa: E402
from pretrain.config import TrainConfig  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.step import lm_loss_sum  # noqa: E402
from pretrain.utils import ROOT, memory_mb, pick_device, seed_everything, synchronize  # noqa: E402


def measure(model, x, y, device, steps: int) -> tuple[float, float, float]:
    """Returns (MiB held after forward, saved-tensor MiB, median forward+backward seconds)."""
    held, times = [], []
    for _ in range(steps):
        gc.collect()
        base = memory_mb(device)
        synchronize(device)
        t = time.perf_counter()
        logits = model(x).logits
        loss = lm_loss_sum(logits, y) / y.numel()
        synchronize(device)
        held.append(memory_mb(device) - base)
        loss.backward()
        synchronize(device)
        times.append(time.perf_counter() - t)
        model.zero_grad(set_to_none=True)
        del logits, loss
    saved = saved_tensor_bytes(lambda: lm_loss_sum(model(x).logits, y)) / 2**20
    return max(held), saved, statistics.median(times[1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/tinystories.yaml"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(2)
    cfg = TrainConfig.from_yaml(args.config)
    device = pick_device(args.device)
    x = torch.randint(0, cfg.model.vocab_size, (cfg.micro_batch_size, cfg.seq_len), device=device)
    y = torch.roll(x, -1, 1)
    print(f"device {device}, batch ({x.size(0)}, {x.size(1)}), {cfg.model.num_layers} layers, fp32")
    print(f"{'mode':<22}{'held after forward':>20}{'autograd saved':>16}{'step ms':>10}")
    rows = {}
    for name in ("plain", "checkpointed"):
        seed_everything(0)
        model = build_model(cfg.model).to(device)
        if name == "checkpointed":
            apply_activation_checkpointing(model)
        held, saved, sec = measure(model, x, y, device, args.steps)
        rows[name] = (held, saved, sec)
        print(f"{name:<22}{held:>17.1f} MiB{saved:>12.1f} MiB{1000 * sec:>10.0f}")
        del model
    p, c = rows["plain"], rows["checkpointed"]
    print(f"memory held {c[0] / max(p[0], 1e-9):.2f}x, step time {c[2] / p[2]:.2f}x")


if __name__ == "__main__":
    main()
