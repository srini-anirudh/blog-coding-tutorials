"""FSDP2 demo: same training as DDP, one shard of the weights and optimizer state per rank.

    GPU, the real use:  torchrun --nproc_per_node 4 c05-pretraining/scripts/fsdp_train.py --steps 100 --hidden-size 1024 --num-layers 12
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


def run(wrap, a: ModelArgs, info, device, steps: int) -> tuple[nn.Module, torch.optim.Optimizer, list[float]]:
    """The same seeded training steps under either wrapper."""
    seed_everything(0)
    model = wrap(build_model(a).to(device))
    opt = build_optimizer(model, TrainConfig(model=a, lr=1e-3), device)
    losses = []
    for step in range(steps):
        g = torch.Generator().manual_seed(1000 * step + info.rank)
        x = torch.randint(0, a.vocab_size, (4, a.max_seq_len), generator=g)
        losses.append(train_step(model, opt, None, [(x, torch.roll(x, -1, 1))], device, "fp32", 1.0).loss)
    return model, opt, losses


def peak_gpu_text(device: torch.device) -> str:
    """Peak tensor memory since the last reset. Only a GPU has this counter."""
    if device.type != "cuda":
        return ""
    text = f", peak {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB"
    torch.cuda.reset_peak_memory_stats()
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(2)
    info = init_distributed()
    device = torch.device(f"cuda:{info.local_rank}" if torch.cuda.is_available() else "cpu")
    a = ModelArgs(hidden_size=args.hidden_size, num_layers=args.num_layers, num_heads=4, num_kv_heads=2,
                  intermediate_size=args.hidden_size * 8 // 3, max_seq_len=64)

    ddp, ddp_opt, ddp_losses = run(lambda m: wrap_ddp(m, info, device), a, info, device, args.steps)
    ddp_text = (f"DDP params {local_param_bytes(ddp) / 2**20:.2f} MiB, optimizer state "
                f"{optimizer_state_bytes(ddp_opt) / 2**20:.2f} MiB{peak_gpu_text(device)}")
    ddp_weights = [p.detach().clone() for p in ddp.parameters()]
    del ddp, ddp_opt  # free the replica so FSDP's peak does not include it
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    fsdp, fsdp_opt, fsdp_losses = run(wrap_fsdp, a, info, device, args.steps)
    fsdp_text = (f"FSDP params {local_param_bytes(fsdp) / 2**20:.2f} MiB, optimizer state "
                 f"{optimizer_state_bytes(fsdp_opt) / 2**20:.2f} MiB{peak_gpu_text(device)}")

    # Gather the shards back into full tensors to compare with DDP's replica.
    gap = max((p.full_tensor() - w).abs().max().item() for p, w in zip(fsdp.parameters(), ddp_weights))
    print(f"rank {info.rank}: {ddp_text}; {fsdp_text}; last loss DDP {ddp_losses[-1]:.6f} "
          f"FSDP {fsdp_losses[-1]:.6f}; max weight gap {gap:.2e}\n", end="", flush=True)
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()
