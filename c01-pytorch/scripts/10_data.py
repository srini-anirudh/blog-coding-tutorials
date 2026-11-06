"""Dataset, collate and DataLoader, ours beside torch's.

Run: python c01-pytorch/scripts/10_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from scratchnet.data import SimpleDataLoader, SpiralDataset, VarLenDataset, collate_dicts, pad_collate


def main() -> None:
    print("== a Dataset answers two questions ==")
    ds = SpiralDataset(points_per_class=10)
    print("len(ds):", len(ds))
    example = ds[0]
    print("ds[0]:", {k: (list(v.shape), v.dtype) for k, v in example.items()})

    print("\n== batching: a list of examples -> one dict of stacked tensors ==")
    batch = collate_dicts([ds[0], ds[1], ds[2]])
    print("collate_dicts of 3:", {k: list(v.shape) for k, v in batch.items()})

    print("\n== shuffle: only the index order changes ==")
    gen = torch.Generator().manual_seed(0)
    plain = SimpleDataLoader(ds, batch_size=4)
    mixed = SimpleDataLoader(ds, batch_size=4, shuffle=True, generator=gen)
    print("indices, no shuffle:", next(plain._index_batches()))
    print("indices, epoch 1   :", next(mixed._index_batches()))
    print("indices, epoch 2   :", next(mixed._index_batches()))

    print("\n== ours against torch.utils.data.DataLoader ==")
    ref = list(DataLoader(ds, batch_size=4, collate_fn=collate_dicts))
    for workers in (0, 2):
        ours = list(SimpleDataLoader(ds, batch_size=4, num_workers=workers))
        same = len(ours) == len(ref) and all(torch.equal(a["x"], b["x"]) for a, b in zip(ours, ref))
        print(f"num_workers={workers}: {len(ours)} batches, same batches as torch: {same}")

    print("\n== ragged sequences need their own collate ==")
    var = VarLenDataset(n=8)
    print("lengths:", [len(var[i]["input_ids"]) for i in range(4)])
    try:
        next(iter(DataLoader(var, batch_size=4)))
    except RuntimeError as err:
        print("default collate ->", str(err).splitlines()[0])
    batch = next(iter(SimpleDataLoader(var, batch_size=4, collate_fn=pad_collate)))
    print("pad_collate:", {k: list(v.shape) for k, v in batch.items()})
    print("input_ids:\n", batch["input_ids"])
    print("attention_mask:\n", batch["attention_mask"])


if __name__ == "__main__":  # required: spawned workers (macOS, Windows) start by re-importing this file
    main()
