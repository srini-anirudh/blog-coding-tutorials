"""Token shards: the corpus as flat uint16 files, one EOS after every document.

Layout on disk (one directory per split):
    train_000.bin, train_001.bin, ...   uint16 token ids, documents back to back
    train_doc_lens.npy                  length of each document including its EOS
    meta.json                           tokenizer, eos id, token and document counts

Flat binary shards are what large pretraining runs use too: they are memory-mapped,
so a loader reads only the windows it needs and many workers can share one file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pretrain.tokenizer import TextTokenizer


def tokenize_documents(
    docs: list[str], tok: TextTokenizer, batch_size: int = 2048
) -> list[np.ndarray]:
    """Encode each document and append EOS. Returns one uint16 array per document."""
    assert tok.vocab_size <= 2**16, "uint16 shards need vocab_size <= 65536"
    out: list[np.ndarray] = []
    for i in range(0, len(docs), batch_size):
        for ids in tok.encode_batch(docs[i : i + batch_size]):
            out.append(np.asarray(ids + [tok.eos_id], dtype=np.uint16))
    return out


def write_shards(
    docs: list[str],
    tok: TextTokenizer,
    out_dir: str | Path,
    split: str,
    shard_tokens: int = 8_000_000,
) -> dict:
    """Tokenize docs and write them as shards of about `shard_tokens` tokens each.

    A shard always ends on a document boundary, so no document is split across files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded = tokenize_documents(docs, tok)
    doc_lens = np.array([len(d) for d in encoded], dtype=np.int32)

    shards, current, count = [], [], 0
    for d in encoded:
        current.append(d)
        count += len(d)
        if count >= shard_tokens:
            shards.append(np.concatenate(current))
            current, count = [], 0
    if current:
        shards.append(np.concatenate(current))

    names = []
    for k, arr in enumerate(shards):
        name = f"{split}_{k:03d}.bin"
        arr.tofile(out_dir / name)
        names.append({"file": name, "tokens": int(arr.size)})
    np.save(out_dir / f"{split}_doc_lens.npy", doc_lens)

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"tokenizer": tok.name, "eos_id": tok.eos_id, "vocab_size": tok.vocab_size})
    meta[split] = {"shards": names, "documents": len(encoded), "tokens": int(doc_lens.sum())}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


class TokenShards:
    """Read-only, memory-mapped view of one split's shards."""

    def __init__(self, data_dir: str | Path, split: str):
        self.dir = Path(data_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.eos_id: int = self.meta["eos_id"]
        self.vocab_size: int = self.meta["vocab_size"]
        self.shards = [
            np.memmap(self.dir / s["file"], dtype=np.uint16, mode="r")
            for s in self.meta[split]["shards"]
        ]

    @property
    def num_tokens(self) -> int:
        return sum(len(s) for s in self.shards)

    def doc_lens(self, split: str) -> np.ndarray:
        return np.load(self.dir / f"{split}_doc_lens.npy")
