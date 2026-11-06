"""Loader: shuffling, rank sharding, worker sharding, deterministic resume."""

import itertools

import torch
from torch.utils.data import DataLoader

from pretrain.loader import EpochSampler, PackedWindows, ResumableLoader
from pretrain.shards import TokenShards
from pretrain.streaming import StreamingWindows


def test_window_targets_are_inputs_shifted(tiny_data):
    ds = PackedWindows(TokenShards(tiny_data, "train"), seq_len=16)
    x, y = ds[3]
    assert x.shape == y.shape == (16,) and x.dtype == torch.long
    assert torch.equal(y[:-1], x[1:])
    x_next, _ = ds[4]
    assert y[-1] == x_next[0]  # windows in one shard overlap by one token
    assert len(ds) == sum((len(s) - 1) // 16 for s in TokenShards(tiny_data, "train").shards)


def test_ranks_get_disjoint_shards_of_one_permutation():
    a = EpochSampler(103, seed=7, epoch=0, rank=0, world_size=2).indices
    b = EpochSampler(103, seed=7, epoch=0, rank=1, world_size=2).indices
    assert len(a) == len(b) == 51 and not set(a) & set(b)
    assert EpochSampler(103, 7, 1).indices != EpochSampler(103, 7, 0).indices  # reshuffled per epoch


def test_resume_mid_epoch_reproduces_batches(tiny_data):
    ds = PackedWindows(TokenShards(tiny_data, "train"), seq_len=64)
    bs = len(ds) // 25  # 25 batches per epoch, so 40 batches cross an epoch boundary
    ref = ResumableLoader(ds, batch_size=bs, seed=3)
    expected = [x for x, _ in itertools.islice(ref, 40)]

    first = ResumableLoader(ds, batch_size=bs, seed=3)
    seen = [x for x, _ in itertools.islice(first, 17)]
    state = first.state_dict()
    second = ResumableLoader(ds, batch_size=bs, seed=3)
    second.load_state_dict(state)
    seen += [x for x, _ in itertools.islice(second, 23)]
    assert second.epoch >= 1  # the 40 batches span more than one epoch
    assert all(torch.equal(a, b) for a, b in zip(seen, expected))


def test_resume_with_workers_and_prefetch(tiny_data):
    ds = PackedWindows(TokenShards(tiny_data, "train"), seq_len=16)
    ref = [x for x, _ in itertools.islice(ResumableLoader(ds, 4, seed=5), 10)]
    a = ResumableLoader(ds, 4, seed=5, num_workers=2, prefetch_factor=4)
    got = [x for x, _ in itertools.islice(a, 4)]  # workers have prefetched further than this
    b = ResumableLoader(ds, 4, seed=5, num_workers=2)
    b.load_state_dict(a.state_dict())
    got += [x for x, _ in itertools.islice(b, 6)]
    assert all(torch.equal(p, q) for p, q in zip(got, ref))


def test_iterable_worker_sharding_covers_each_window_once(tiny_data):
    shards = TokenShards(tiny_data, "train")
    full = [x for x, _ in StreamingWindows(shards, 16)]
    got = []
    for rank in range(2):
        dl = DataLoader(StreamingWindows(shards, 16, rank=rank, world_size=2), batch_size=None, num_workers=2)
        got += [x for x in (b[0] for b in dl)]
    key = lambda t: tuple(t.tolist())  # noqa: E731
    assert sorted(map(key, got)) == sorted(map(key, full))
