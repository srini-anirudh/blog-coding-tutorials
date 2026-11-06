"""FSDP2 demo: same training as DDP, one shard of the weights and optimizer state per rank.

    GPU, the real use:  torchrun --nproc_per_node 4 c05-pretraining/scripts/fsdp_train.py --steps 100
    CPU, for shapes:    torchrun --nproc_per_node 2 c05-pretraining/scripts/fsdp_train.py --steps 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402

from pretrain.config import ModelArgs, TrainConfig  # noqa: E402
from pretrain.distributed import cleanup, init_distributed, wrap_ddp  # noqa: E402
from pretrain.fsdp import local_param_bytes, wrap_fsdp  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.optim import build_optimizer  # noqa: E402
from pretrain.step import train_step  # noqa: E402
from pretrain.utils import seed_everything  # noqa: E402
from pretrain.zero import optimizer_state_bytes  # noqa: E402


ARGS = ModelArgs(hidden_size=128, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=344, max_seq_len=64)


def run(wrap, info, device, steps: int) -> tuple[nn.Module, torch.optim.Optimizer, list[float]]:
    """The same seeded training steps under either wrapper."""
    seed_everything(0)
    model = wrap(build_model(ARGS).to(device))
    opt = build_optimizer(model, TrainConfig(model=ARGS, lr=1e-3), device)
    losses = []
    for step in range(steps):
        g = torch.Generator().manual_seed(1000 * step + info.rank)
        x = torch.randint(0, ARGS.vocab_size, (4, 64), generator=g)
        losses.append(train_step(model, opt, None, [(x, torch.roll(x, -1, 1))], device, "fp32", 1.0).loss)
    return model, opt, losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(2)
    info = init_distributed()
    device = torch.device(f"cuda:{info.local_rank}" if torch.cuda.is_available() else "cpu")

    ddp, ddp_opt, ddp_losses = run(lambda m: wrap_ddp(m, info, device), info, device, args.steps)
    fsdp, fsdp_opt, fsdp_losses = run(wrap_fsdp, info, device, args.steps)

    # Gather the shards back into full tensors to compare with DDP's replica.
    gap = max((a.full_tensor() - b).abs().max().item()
              for a, b in zip(fsdp.parameters(), ddp.parameters()))
    print(f"rank {info.rank}: params DDP {local_param_bytes(ddp) / 2**20:.2f} MiB, "
          f"FSDP {local_param_bytes(fsdp) / 2**20:.2f} MiB; optimizer state DDP "
          f"{optimizer_state_bytes(ddp_opt) / 2**20:.2f} MiB, FSDP {optimizer_state_bytes(fsdp_opt) / 2**20:.2f} MiB; "
          f"last loss DDP {ddp_losses[-1]:.6f} FSDP {fsdp_losses[-1]:.6f}; max weight gap {gap:.2e}", flush=True)
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()
