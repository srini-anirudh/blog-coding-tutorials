"""Gradient accumulation against one big batch, on real tokens, in fp32 on CPU.

Prints the largest weight difference after one SGD step for each way of accumulating.

    python c05-pretraining/scripts/check_accumulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pretrain.config import ModelArgs  # noqa: E402
from pretrain.loader import PackedWindows  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.packing import IGNORE  # noqa: E402
from pretrain.shards import TokenShards  # noqa: E402
from pretrain.step import train_step  # noqa: E402
from pretrain.utils import resolve  # noqa: E402

A = ModelArgs(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=172, max_seq_len=64)
CPU = torch.device("cpu")


def fresh():
    torch.manual_seed(0)
    return build_model(A)


def weights(m) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def ours(micro) -> torch.Tensor:
    m = fresh()
    train_step(m, torch.optim.SGD(m.parameters(), lr=0.1), None, micro, CPU, "fp32", float("inf"))
    return weights(m)


def mean_of_means(micro) -> torch.Tensor:
    m = fresh()
    for x, y in micro:
        loss = F.cross_entropy(m(x).logits.reshape(-1, A.vocab_size), y.reshape(-1), ignore_index=IGNORE)
        (loss / len(micro)).backward()
    torch.optim.SGD(m.parameters(), lr=0.1).step()
    return weights(m)


def main() -> None:
    ds = PackedWindows(TokenShards(resolve("data_cache/tinystories"), "train"), 64)
    x, y = map(torch.stack, zip(*(ds[i] for i in range(8))))
    split = lambda a, b: [(a[i : i + 2], b[i : i + 2]) for i in range(0, 8, 2)]  # noqa: E731

    big = ours([(x, y)])
    print("packed data, every micro-batch has the same number of targets")
    print(f"  token-weighted accumulation  max |diff| {(ours(split(x, y)) - big).abs().max():.2e}")
    print(f"  mean of means                max |diff| {(mean_of_means(split(x, y)) - big).abs().max():.2e}")

    y2 = y.clone()
    y2[:2, 8:] = IGNORE  # the first micro-batch keeps only 16 of 128 targets, as padding would
    big = ours([(x, y2)])
    print("unequal targets per micro-batch (112 of 512 ignored)")
    print(f"  token-weighted accumulation  max |diff| {(ours(split(x, y2)) - big).abs().max():.2e}")
    print(f"  mean of means                max |diff| {(mean_of_means(split(x, y2)) - big).abs().max():.2e}")


if __name__ == "__main__":
    main()
