"""Data parallelism: one model replica per process, gradients averaged with all-reduce.

torchrun starts the processes and sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR and
MASTER_PORT in each one's environment. The backend is NCCL on CUDA GPUs and gloo on CPU,
which is how this tutorial runs real two-process DDP without a GPU.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def init_distributed(timeout_s: int = 300) -> DistInfo:
    """Join the process group if torchrun launched us; otherwise run single-process."""
    if "RANK" not in os.environ:
        return DistInfo()
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # A finite timeout turns a dead peer into an error instead of a hang forever.
    timeout = datetime.timedelta(seconds=timeout_s)
    attempt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", 0))
    if attempt == 0:
        dist.init_process_group(backend, timeout=timeout)
    else:
        # After a restart torchrun's store still holds the addresses the dead processes published,
        # and ranks would try to connect to those. A new key prefix gives each attempt a clean slate.
        agent_store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]),
                                    is_master=False, timeout=timeout)
        dist.init_process_group(backend, store=dist.PrefixStore(f"attempt_{attempt}", agent_store),
                                rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]),
                                timeout=timeout)
    info = DistInfo(dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size())
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
    return info


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(model: nn.Module, info: DistInfo, device: torch.device) -> nn.Module:
    """DDP broadcasts rank 0's weights at construction, then all-reduces grads in buckets
    while backward is still running, so communication overlaps compute."""
    if not info.enabled:
        return model
    ids = [info.local_rank] if device.type == "cuda" else None
    return nn.parallel.DistributedDataParallel(model, device_ids=ids)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def allreduce_gradients_by_hand(model: nn.Module) -> None:
    """What DDP does, without the bucketing and overlap: average every gradient."""
    world = dist.get_world_size()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world


def mean_across_ranks(value: float, device: torch.device) -> float:
    """Average a scalar metric (loss, tokens/s) across ranks for logging."""
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / dist.get_world_size()
