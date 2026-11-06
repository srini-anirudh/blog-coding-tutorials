"""A resumable, rank-sharded loader over packed windows.

The dataset is map-style: window i is a fixed slice of the token shards, so it can be
read by index from any process. Order comes from a permutation that depends only on
(seed, epoch). The loader counts the batches it has handed to the training loop.
Those three numbers (seed, epoch, batches consumed) are the whole data position: saving
them in a checkpoint lets a resumed run see exactly the batches it would have seen.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from pretrain.shards import TokenShards


class PackedWindows(Dataset):
    """Window i -> (input_ids, targets), both (L,) int64, targets shifted by one.

    Windows never cross a shard file. Each shard of n tokens gives (n - 1) // L windows.
    """

    def __init__(self, shards: TokenShards, seq_len: int):
        self.shards = shards.shards
        self.seq_len = seq_len
        counts = [(len(s) - 1) // seq_len for s in self.shards]
        self.offsets = np.cumsum([0] + counts)  # first global window index of each shard

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = int(np.searchsorted(self.offsets, i, side="right") - 1)
        start = (i - self.offsets[s]) * self.seq_len
        chunk = torch.from_numpy(self.shards[s][start : start + self.seq_len + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


class EpochSampler(Sampler[int]):
    """Shuffled indices for one (epoch, rank), starting after `skip` samples.

    Every rank builds the same permutation from (seed, epoch) and takes every
    world_size-th entry, so ranks see disjoint windows without talking to each other.
    """

    def __init__(self, n: int, seed: int, epoch: int, rank: int = 0, world_size: int = 1, skip: int = 0):
        g = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(n, generator=g)
        perm = perm[: n - n % world_size]  # equal share per rank
        self.indices = perm[rank::world_size][skip:].tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class ResumableLoader:
    """Endless iterator of (x, y) batches with a saveable position."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
    ):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.rank, self.world_size = rank, world_size
        self.loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
        if num_workers > 0:
            # Workers prepare `prefetch_factor` batches each while the GPU is busy.
            self.loader_kwargs |= dict(prefetch_factor=prefetch_factor, persistent_workers=False)
        self.epoch, self.consumed = 0, 0
        self._it = None

    def _start_epoch(self) -> None:
        sampler = EpochSampler(
            len(self.dataset), self.seed, self.epoch, self.rank, self.world_size,
            skip=self.consumed * self.batch_size,
        )
        dl = DataLoader(self.dataset, self.batch_size, sampler=sampler, drop_last=True, **self.loader_kwargs)
        self._it = iter(dl)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._it is None:
            self._start_epoch()
        try:
            batch = next(self._it)
        except StopIteration:
            self.epoch, self.consumed = self.epoch + 1, 0
            self._start_epoch()
            batch = next(self._it)
        # Count in the main process, when the batch reaches the training loop. Batches
        # that workers prefetched but we never used are not counted, so they are not lost.
        self.consumed += 1
        return batch

    def state_dict(self) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "consumed": self.consumed,
                "rank": self.rank, "world_size": self.world_size}

    def load_state_dict(self, state: dict) -> None:
        assert state["seed"] == self.seed, "data seed changed between runs"
        assert state["world_size"] == self.world_size, "resharding on resume is not supported"
        self.epoch, self.consumed = state["epoch"], state["consumed"]
        self._it = None  # rebuilt lazily from the restored position
