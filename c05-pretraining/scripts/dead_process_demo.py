"""Kill one rank mid-run. torchrun restarts both, and training resumes from the checkpoint.

    torchrun --nproc_per_node 2 --max-restarts 1 c05-pretraining/scripts/dead_process_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from pretrain.config import TrainConfig  # noqa: E402
from pretrain.distributed import cleanup, init_distributed  # noqa: E402
from pretrain.trainer import train  # noqa: E402
from pretrain.utils import resolve  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    attempt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", 0))
    info = init_distributed(timeout_s=60)
    cfg = TrainConfig.from_yaml(resolve("configs/smoke.yaml"), max_steps=30, ckpt_every=10, log_every=10,
                                eval_every=10_000, out_dir="runs/dead_process")
    print(f"attempt {attempt} rank {info.rank}: starting", flush=True)
    if attempt == 0:
        train(cfg, info, resume=False, stop_after=15)
        if info.rank == 1:
            print("rank 1: simulated crash at step 15", flush=True)
            os._exit(1)
        dist.barrier()  # raises when the peer disappears
    train(cfg, info, resume=True)
    if info.is_main:
        print("finished after restart", flush=True)
    cleanup()


if __name__ == "__main__":
    main()
