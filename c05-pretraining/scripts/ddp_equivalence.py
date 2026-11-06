"""Prove DDP is the same as one big batch: same weights after one step, on every rank.

    torchrun --nproc_per_node 2 c05-pretraining/scripts/ddp_equivalence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pretrain.config import ModelArgs  # noqa: E402
from pretrain.distributed import allreduce_gradients_by_hand, cleanup, init_distributed, wrap_ddp  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.step import train_step  # noqa: E402

A = ModelArgs(vocab_size=300, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
              intermediate_size=64, max_seq_len=32)
CPU = torch.device("cpu")


def fresh():
    torch.manual_seed(0)
    return build_model(A)


def weights(m) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the result as JSON (used by the test)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    info = init_distributed()
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, A.vocab_size, (8, 32), generator=g)  # the same global batch on every rank
    y = torch.roll(x, -1, 1)
    xl, yl = x[info.rank :: info.world_size], y[info.rank :: info.world_size]  # (4, 32) on each rank

    single = fresh()
    train_step(single, torch.optim.SGD(single.parameters(), lr=0.1), None, [(x, y)], CPU, "fp32", float("inf"))

    ddp = wrap_ddp(fresh(), info, CPU)
    train_step(ddp, torch.optim.SGD(ddp.parameters(), lr=0.1), None, [(xl, yl)], CPU, "fp32", float("inf"))

    hand = fresh()
    F.cross_entropy(hand(xl).logits.reshape(-1, A.vocab_size), yl.reshape(-1)).backward()
    allreduce_gradients_by_hand(hand)
    torch.optim.SGD(hand.parameters(), lr=0.1).step()

    w_single, w_ddp, w_hand = weights(single), weights(ddp), weights(hand)
    gathered = [torch.zeros_like(w_ddp) for _ in range(info.world_size)]
    dist.all_gather(gathered, w_ddp)
    result = {
        "ddp_vs_single": (w_ddp - w_single).abs().max().item(),
        "hand_vs_ddp": (w_hand - w_ddp).abs().max().item(),
        "ranks_identical": all(torch.equal(g_, gathered[0]) for g_ in gathered),
        "local_batch": list(xl.shape),
        "global_batch": list(x.shape),
    }
    if info.is_main:
        print(f"world size {info.world_size}, backend {dist.get_backend()}, "
              f"local batch {tuple(xl.shape)}, global batch {tuple(x.shape)}")
        print(f"max |DDP - single process big batch|  {result['ddp_vs_single']:.2e}")
        print(f"max |all-reduce by hand - DDP|        {result['hand_vs_ddp']:.2e}")
        print(f"weights identical on every rank:      {result['ranks_identical']}")
        if args.out:
            Path(args.out).write_text(json.dumps(result))
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()
