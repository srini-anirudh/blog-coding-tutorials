"""Tokenize the full TinyStories train file into uint16 shards for the scaling sweep.

The sweep's largest runs read about 400M tokens, so the 128 MiB slice C5 uses (33M tokens) is too small:
those runs would repeat data, and repetition is a separate experiment (06_repetition.py). This script reuses
C5's download, tokenizer and shard writer, and points them at this lab's own data_cache/.

    python labs/scaling-laws-from-first-principles/scripts/00_prepare_data.py            # whole file, ~2.2 GB
    python labs/scaling-laws-from-first-principles/scripts/00_prepare_data.py --train-mb 64   # quick check
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
C5 = LAB.parents[1] / "c05-pretraining"
sys.path.insert(0, str(C5))

from pretrain.corpus import load_documents  # noqa: E402  (C5)
from pretrain.shards import write_shards  # noqa: E402  (C5)
from pretrain.tokenizer import load_file, train_bpe  # noqa: E402  (C5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(LAB / "data_cache" / "tinystories"))
    ap.add_argument("--train-mb", type=int, default=2300, help="first N MiB of the 2.2 GB train file")
    args = ap.parse_args()
    out = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_docs = load_documents("train", out, train_bytes=args.train_mb * 2**20)
    val_docs = load_documents("val", out)
    print(f"documents: train {len(train_docs):,}  val {len(val_docs):,}  ({time.time() - t0:.0f}s)")

    # Same 4096-token BPE as C5. Copy C5's file when it exists so token ids match across folders.
    tok_path = out / "bpe4096.json"
    c5_tok = C5 / "data_cache" / "tinystories" / "bpe4096.json"
    if not tok_path.exists() and c5_tok.exists():
        shutil.copy(c5_tok, tok_path)
    if tok_path.exists():
        tok = load_file(tok_path)
    else:
        tok = train_bpe(train_docs[:100_000], 4096)
        tok.save(tok_path)
    tok.name = "bpe-4096"

    t0 = time.time()
    for split, docs in (("train", train_docs), ("val", val_docs)):
        meta = write_shards(docs, tok, out, split, shard_tokens=50_000_000)
        print(f"{split}: {meta[split]['tokens']:,} tokens in {len(meta[split]['shards'])} shard(s)")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
