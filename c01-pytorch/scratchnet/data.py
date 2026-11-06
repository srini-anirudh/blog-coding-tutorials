"""Datasets and collate functions.

A Dataset answers two questions: how many examples (`__len__`) and what is
example i (`__getitem__`). A DataLoader picks indices, fetches examples,
possibly in worker processes, and calls `collate_fn` to stack them into a batch.
"""

from __future__ import annotations

import itertools
import math
import time
import traceback
from collections.abc import Callable, Iterator

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset


class SpiralDataset(Dataset):
    """K interleaved spiral arms in 2D. A linear model cannot separate them.

    Example i: {"x": FloatTensor [2], "y": LongTensor [] (class id)}.
    """

    def __init__(self, points_per_class: int = 300, num_classes: int = 3, noise: float = 0.25, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        xs, ys = [], []
        for k in range(num_classes):
            r = torch.linspace(0.05, 1.0, points_per_class)
            # Each arm sweeps 7 radians (just over one turn), rotated 2pi/K from the previous arm.
            theta = torch.linspace(0, 7, points_per_class) + k * 2 * math.pi / num_classes
            theta = theta + noise * torch.randn(points_per_class, generator=g)
            xs.append(torch.stack([r * torch.sin(theta), r * torch.cos(theta)], dim=1))
            ys.append(torch.full((points_per_class,), k, dtype=torch.long))
        self.x = torch.cat(xs)  # [N, 2]
        self.y = torch.cat(ys)  # [N]
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[i], "y": self.y[i]}


def collate_dicts(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """[{"x": [2], "y": []}, ...] of length B -> {"x": [B, 2], "y": [B]}.

    This is what the default collate does for dicts of equal-shape tensors,
    written out so the step is visible.
    """
    return {key: torch.stack([example[key] for example in batch]) for key in batch[0]}


def _worker_loop(dataset: Dataset, collate_fn: Callable, tasks: mp.Queue, results: mp.Queue) -> None:
    """One worker process: take (batch_id, indices), build that batch, send it back."""
    torch.set_num_threads(1)  # N workers with M threads each would oversubscribe the cores
    while (task := tasks.get()) is not None:
        batch_id, indices = task
        try:
            results.put((batch_id, collate_fn([dataset[i] for i in indices]), None))
        except Exception:
            results.put((batch_id, None, traceback.format_exc()))


class SimpleDataLoader:
    """The core of torch.utils.data.DataLoader in one class.

    shuffle picks the index order once per epoch. batch_size cuts that order
    into lists of indices. collate_fn turns each list of examples into one batch.

    With num_workers > 0, separate processes build the batches. Workers finish
    in any order, so finished batches wait in `ready` until their turn comes.
    At most num_workers * prefetch_factor batches are in flight: that bound is
    the prefetch, and it is also what stops workers from filling all of memory.

    Workers start fresh every epoch. torch's version also watches for workers
    that die and seeds each one, so random augmentation differs between workers.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn: Callable = collate_dicts,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.collate_fn, self.num_workers, self.prefetch_factor = collate_fn, num_workers, prefetch_factor
        self.pin_memory, self.drop_last, self.generator = pin_memory, drop_last, generator

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def _index_batches(self) -> Iterator[list[int]]:
        n = len(self.dataset)
        order = torch.randperm(n, generator=self.generator) if self.shuffle else torch.arange(n)
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            yield order[start : start + self.batch_size].tolist()

    def _finish(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Pinned pages cannot be swapped out, so the GPU can copy from them without a staging buffer.
        return {k: v.pin_memory() for k, v in batch.items()} if self.pin_memory else batch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.num_workers == 0:
            for indices in self._index_batches():
                yield self._finish(self.collate_fn([self.dataset[i] for i in indices]))
        else:
            yield from self._iter_workers()

    def _iter_workers(self) -> Iterator[dict[str, torch.Tensor]]:
        results: mp.Queue = mp.Queue()
        queues = [mp.Queue() for _ in range(self.num_workers)]
        workers = [mp.Process(target=_worker_loop, args=(self.dataset, self.collate_fn, q, results), daemon=True)
                   for q in queues]
        for w in workers:
            w.start()

        todo = enumerate(self._index_batches())
        turn = itertools.cycle(queues)  # hand out work round robin

        def send_next() -> int:
            task = next(todo, None)
            if task is None:
                return 0
            next(turn).put(task)
            return 1

        in_flight = sum(send_next() for _ in range(self.num_workers * self.prefetch_factor))
        ready: dict[int, dict[str, torch.Tensor]] = {}
        next_id = 0
        try:
            while in_flight:
                while next_id not in ready:
                    batch_id, batch, error = results.get()
                    if error is not None:
                        raise RuntimeError(f"worker failed on batch {batch_id}:\n{error}")
                    ready[batch_id] = batch
                yield self._finish(ready.pop(next_id))
                next_id += 1
                in_flight += send_next() - 1
        finally:
            # Also runs when the caller breaks out of the loop early.
            for q in queues:
                q.put(None)
            for w in workers:
                w.join(timeout=5)
                if w.is_alive():
                    w.terminate()


class VarLenDataset(Dataset):
    """Token sequences of different lengths, the shape of every language dataset.

    Example i: {"input_ids": LongTensor [L_i], "label": LongTensor []}.
    The label is 1 when the sequence sum is even; it is only here so the
    batch has something besides the ragged sequences.
    """

    def __init__(self, n: int = 64, min_len: int = 2, max_len: int = 9, vocab: int = 20, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        lengths = torch.randint(min_len, max_len + 1, (n,), generator=g)
        self.seqs = [torch.randint(1, vocab, (int(length),), generator=g) for length in lengths]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        seq = self.seqs[i]
        return {"input_ids": seq, "label": (seq.sum() % 2 == 0).long()}


def pad_collate(batch: list[dict[str, torch.Tensor]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Right-pad ragged sequences to the longest in the batch.

    Returns input_ids [B, T_max], attention_mask [B, T_max] (1 = real token),
    lengths [B], label [B]. torch.stack would fail here: shapes differ.
    """
    lengths = torch.tensor([len(ex["input_ids"]) for ex in batch])
    t_max = int(lengths.max())
    input_ids = torch.full((len(batch), t_max), pad_id, dtype=torch.long)
    for row, ex in enumerate(batch):
        input_ids[row, : len(ex["input_ids"])] = ex["input_ids"]
    # [T_max] < [B, 1] broadcasts to [B, T_max]: position t is real when t < length.
    attention_mask = (torch.arange(t_max)[None, :] < lengths[:, None]).long()
    labels = torch.stack([ex["label"] for ex in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "lengths": lengths, "label": labels}


class SlowDataset(Dataset):
    """Wraps a dataset and burns `delay_s` of CPU time per example.

    Stands in for real per-example work (decoding a JPEG, tokenizing text)
    so the effect of DataLoader workers is measurable.
    """

    def __init__(self, base: Dataset, delay_s: float = 0.001) -> None:
        self.base, self.delay_s = base, delay_s

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        end = time.perf_counter() + self.delay_s
        while time.perf_counter() < end:  # busy-wait: a sleep would not use the CPU
            pass
        return self.base[i]
