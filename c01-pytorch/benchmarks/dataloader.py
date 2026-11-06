"""DataLoader throughput with per-example work: num_workers, persistent workers and prefetch.

Run: python c01-pytorch/benchmarks/dataloader.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from scratchnet.data import SlowDataset, SpiralDataset, collate_dicts

torch.set_num_threads(2)


def epoch_times(loader: DataLoader, epochs: int = 2) -> list[float]:
    times = []
    for _ in range(epochs):
        t0 = time.perf_counter()
        for _batch in loader:
            pass
        times.append(time.perf_counter() - t0)
    return times


def main() -> None:
    ds = SlowDataset(SpiralDataset(points_per_class=200), delay_s=0.002)  # 600 examples x 2 ms = 1.2 s of work
    print(f"{len(ds)} examples, {ds.delay_s * 1e3:.0f} ms of CPU work each, batch 32")
    print(f"{'config':<34} {'epoch 1 s':>10} {'epoch 2 s':>10} {'ex/s (ep 2)':>12}")
    configs = [
        ("num_workers=0", dict(num_workers=0)),
        ("num_workers=2", dict(num_workers=2)),
        ("num_workers=2 persistent", dict(num_workers=2, persistent_workers=True)),
        ("num_workers=4 persistent", dict(num_workers=4, persistent_workers=True)),
        ("num_workers=4 persistent prefetch=4", dict(num_workers=4, persistent_workers=True, prefetch_factor=4)),
    ]
    for name, kwargs in configs:
        loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_dicts, **kwargs)
        t1, t2 = epoch_times(loader)
        print(f"{name:<34} {t1:>10.2f} {t2:>10.2f} {len(ds) / t2:>12.0f}")
        del loader


if __name__ == "__main__":  # required: workers on macOS start by re-importing this file
    main()
