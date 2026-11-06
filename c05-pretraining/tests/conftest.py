"""Shared fixtures: a tiny synthetic corpus, tokenizer and shard directory.

Tests never touch the downloaded TinyStories data, so they run offline in seconds.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from pretrain.config import ModelArgs, TrainConfig
from pretrain.shards import write_shards
from pretrain.tokenizer import train_bpe

NAMES = ["Tom", "Lily", "Ben", "Mia", "Sam"]
THINGS = ["ball", "dog", "red box", "big tree", "kite", "cake"]
VERBS = ["found", "liked", "saw", "lost", "shared"]


def synthetic_docs(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        sents = [f"{rng.choice(NAMES)} {rng.choice(VERBS)} a {rng.choice(THINGS)}."
                 for _ in range(rng.randint(1, 12))]
        docs.append("Once upon a time, " + " ".join(sents) + " The end.")
    return docs


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("tiny_data")
    train_docs, val_docs = synthetic_docs(600, 0), synthetic_docs(100, 1)
    tok = train_bpe(train_docs, vocab_size=300)
    tok.save(out / "tok.json")
    write_shards(train_docs, tok, out, "train", shard_tokens=4_000)  # several shards on purpose
    write_shards(val_docs, tok, out, "val")
    return out


@pytest.fixture
def tiny_cfg(tiny_data: Path, tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        data_dir=str(tiny_data), out_dir=str(tmp_path / "run"), seq_len=32,
        micro_batch_size=4, grad_accum_steps=2,
        model=ModelArgs(vocab_size=300, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                        intermediate_size=64, max_seq_len=32),
        lr=3e-3, min_lr=3e-4, max_steps=8, warmup_steps=2, precision="fp32", device="cpu",
        log_every=1, eval_every=4, eval_batches=1, ckpt_every=4,
    )
