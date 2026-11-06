"""Pretrain a small LLM. Single process, or DDP under torchrun.

    python c05-pretraining/scripts/train.py --config c05-pretraining/configs/tinystories.yaml
    python c05-pretraining/scripts/train.py --config ... --model tinyllm        # C4's model
    torchrun --nproc_per_node 2 c05-pretraining/scripts/train.py --config c05-pretraining/configs/ddp_cpu.yaml

Re-running the same command resumes from the newest valid checkpoint in out_dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pretrain.config import TrainConfig  # noqa: E402
from pretrain.distributed import cleanup, init_distributed  # noqa: E402
from pretrain.mfu import measure_matmul_peak  # noqa: E402
from pretrain.trainer import train  # noqa: E402
from pretrain.utils import pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", dest="model_type", choices=["llama", "tinyllm"])
    ap.add_argument("--out-dir")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--precision", choices=["fp32", "bf16"])
    ap.add_argument("--device")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "fresh", "threads")}
    cfg = TrainConfig.from_yaml(args.config, **overrides)
    info = init_distributed()
    peak = None
    if not info.enabled:
        device = pick_device(cfg.device)
        dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float32
        peak = measure_matmul_peak(device, dtype)
        print(f"measured {dtype} matmul peak on {device}: {peak / 1e12:.2f} TFLOP/s")
    try:
        train(cfg, info, resume=not args.fresh, peak_flops=peak)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
