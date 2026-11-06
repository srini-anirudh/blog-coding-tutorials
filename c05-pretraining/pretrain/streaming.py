"""Worker sharding for an IterableDataset.

A map-style dataset gets worker sharding for free: the DataLoader hands each worker
different indices. An IterableDataset does not. Every worker runs the same __iter__,
so without the split below each of N workers yields the whole stream and the model
sees every batch N times. This class shows the split explicitly, across both
distributed ranks and DataLoader workers.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from pretrain.shards import TokenShards


class StreamingWindows(IterableDataset):
    """Yields (x, y) windows in file order; each (rank, worker) takes every k-th window."""

    def __init__(self, shards: TokenShards, seq_len: int, rank: int = 0, world_size: int = 1):
        self.shards, self.seq_len = shards.shards, seq_len
        self.rank, self.world_size = rank, world_size

    def __iter__(self):
        info = get_worker_info()
        worker, num_workers = (info.id, info.num_workers) if info else (0, 1)
        # One global stream id per (rank, worker) pair.
        stream = self.rank * num_workers + worker
        num_streams = self.world_size * num_workers
        k = 0
        L = self.seq_len
        for shard in self.shards:
            for w in range((len(shard) - 1) // L):
                if k % num_streams == stream:
                    chunk = torch.from_numpy(shard[w * L : w * L + L + 1].astype(np.int64))
                    yield chunk[:-1], chunk[1:]
                k += 1
