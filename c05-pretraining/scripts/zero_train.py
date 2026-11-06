"""ZeRO stage 1 on CPU with gloo: optimizer state per rank, compared with DDP plus plain AdamW.

    torchrun --nproc_per_node 2 c05-pretraining/scripts/zero_train.py

Stages 2 and 3 need DeepSpeed and GPUs: see scripts/deepspeed_train.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from pretrain.config import ModelArgs  # noqa: E402
from pretrain.distributed import cleanup, init_distributed, wrap_ddp  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.step import train_step  # noqa: E402
from pretrain.utils import seed_everything  # noqa: E402
from pretrain.zero import deepspeed_config, optimizer_state_bytes, wrap_zero1  # noqa: E402


def run(zero: bool, steps: int, info) -> tuple[list[float], int, torch.Tensor]:
    seed_everything(0)
    a = ModelArgs(hidden_size=128, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=344, max_seq_len=64)
    device = torch.device("cpu")
    model = wrap_ddp(build_model(a).to(device), info, device)
    opt = (wrap_zero1(model, 1e-3, 0.1) if zero
           else torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95)))
    losses = []
    for step in range(steps):
        g = torch.Generator().manual_seed(1000 * step + info.rank)
        x = torch.randint(0, a.vocab_size, (4, 64), generator=g)
        losses.append(train_step(model, opt, None, [(x, torch.roll(x, -1, 1))], device, "fp32", 1.0).loss)
    weights = torch.cat([p.detach().flatten() for p in model.parameters()])
    return losses, optimizer_state_bytes(opt), weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(2)
    info = init_distributed()
    plain_losses, plain_bytes, plain_w = run(False, args.steps, info)
    zero_losses, zero_bytes, zero_w = run(True, args.steps, info)
    diff = (plain_w - zero_w).abs().max().item()
    dist.barrier()
    print(f"rank {info.rank}: optimizer state DDP+AdamW {plain_bytes / 2**20:.2f} MiB, "
          f"ZeRO-1 {zero_bytes / 2**20:.2f} MiB, final-loss gap "
          f"{abs(plain_losses[-1] - zero_losses[-1]):.2e}, max weight gap {diff:.2e}\n", end="", flush=True)
    if info.is_main:
        path = Path(__file__).resolve().parents[1] / "configs/deepspeed_zero3.json"
        path.write_text(json.dumps(deepspeed_config(3, 16, 2, 3e-3, 1.0), indent=2) + "\n")
        print(f"wrote {path.name}. Stages 2 and 3 are GPU only: scripts/deepspeed_train.py --stage 3")
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()
