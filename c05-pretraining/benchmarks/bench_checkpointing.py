"""Activation checkpointing: memory against step time.

Each mode runs in a fresh process, because a process's peak memory only ever goes up and the
allocator keeps freed blocks for reuse. Two runs in one process would measure each other.

    python c05-pretraining/benchmarks/bench_checkpointing.py --device cuda
    python c05-pretraining/benchmarks/bench_checkpointing.py --model tinyllm
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pretrain.activation_ckpt import apply_activation_checkpointing, saved_tensor_bytes  # noqa: E402
from pretrain.config import TrainConfig  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.step import lm_loss_sum  # noqa: E402
from pretrain.utils import ROOT, peak_rss_mb, pick_device, seed_everything, synchronize  # noqa: E402


def peak_mb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else peak_rss_mb()


def measure_one(args: argparse.Namespace) -> dict:
    """Runs in the child process. Returns memory growth and median step time for one mode."""
    cfg = TrainConfig.from_yaml(args.config)
    device = pick_device(args.device)
    seed_everything(0)
    model = build_model(cfg.model, args.model).to(device)
    if args.mode == "checkpointed":
        apply_activation_checkpointing(model)
    x = torch.randint(0, cfg.model.vocab_size, (cfg.micro_batch_size, cfg.seq_len), device=device)
    y = torch.roll(x, -1, 1)
    saved = saved_tensor_bytes(lambda: lm_loss_sum(model(x).logits, y)) / 2**20  # graph dropped on return

    synchronize(device)
    base, times = peak_mb(device), []
    for _ in range(args.steps):
        t = time.perf_counter()
        (lm_loss_sum(model(x).logits, y) / y.numel()).backward()
        synchronize(device)
        times.append(time.perf_counter() - t)
        model.zero_grad(set_to_none=True)
    return {"saved": saved, "peak_growth": peak_mb(device) - base, "sec": statistics.median(times[1:])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/tinystories.yaml"))
    ap.add_argument("--model", default="llama", choices=["llama", "tinyllm"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--mode", choices=["plain", "checkpointed"], help="internal: measure one mode")
    args = ap.parse_args()
    if args.mode:
        print(json.dumps(measure_one(args)))
        return

    cfg = TrainConfig.from_yaml(args.config)
    print(f"model {args.model}, device {pick_device(args.device)}, batch ({cfg.micro_batch_size}, {cfg.seq_len}), "
          f"{cfg.model.num_layers} layers, fp32, one fresh process per mode")
    print(f"{'mode':<16}{'autograd saved':>16}{'peak growth':>14}{'step ms':>10}")
    rows = {}
    for mode in ("plain", "checkpointed"):
        out = subprocess.run([sys.executable, __file__, "--config", args.config, "--model", args.model,
                              "--device", args.device, "--steps", str(args.steps), "--mode", mode],
                             check=True, capture_output=True, text=True).stdout
        r = rows[mode] = json.loads(out.splitlines()[-1])
        print(f"{mode:<16}{r['saved']:>12.1f} MiB{r['peak_growth']:>10.1f} MiB{1000 * r['sec']:>10.0f}")
    p, c = rows["plain"], rows["checkpointed"]
    print(f"autograd saved {c['saved'] / p['saved']:.2f}x, peak growth {c['peak_growth'] / p['peak_growth']:.2f}x, "
          f"step time {c['sec'] / p['sec']:.2f}x")


if __name__ == "__main__":
    main()
