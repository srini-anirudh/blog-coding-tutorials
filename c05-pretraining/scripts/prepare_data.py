"""Download TinyStories, train (or load) a tokenizer, and write EOS-separated token shards.

    python c05-pretraining/scripts/prepare_data.py                    # 4096-token BPE (default)
    python c05-pretraining/scripts/prepare_data.py --tokenizer gpt2   # GPT-2's tokenizer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pretrain.corpus import load_documents  # noqa: E402
from pretrain.shards import write_shards  # noqa: E402
from pretrain.tokenizer import load_file, load_pretrained, train_bpe  # noqa: E402
from pretrain.utils import resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_cache/tinystories")
    ap.add_argument("--train-mb", type=int, default=128, help="first N MiB of the 2.2 GB train file")
    ap.add_argument("--tokenizer", choices=["bpe", "gpt2"], default="bpe")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--tokenizer-docs", type=int, default=100_000, help="documents used to train BPE")
    ap.add_argument("--shard-tokens", type=int, default=8_000_000)
    args = ap.parse_args()
    data_dir = resolve(args.data_dir)

    t0 = time.time()
    train_docs = load_documents("train", data_dir, train_bytes=args.train_mb * 2**20)
    val_docs = load_documents("val", data_dir)
    print(f"documents: train {len(train_docs):,}  val {len(val_docs):,}  ({time.time() - t0:.1f}s)")
    print(f"first document ({len(train_docs[0])} chars): {train_docs[0][:160]!r}...")

    t0 = time.time()
    if args.tokenizer == "gpt2":
        tok = load_pretrained("openai-community/gpt2")
    else:
        tok_path = data_dir / f"bpe{args.vocab_size}.json"
        if tok_path.exists():
            tok = load_file(tok_path)
        else:
            tok = train_bpe(train_docs[: args.tokenizer_docs], args.vocab_size)
            tok.save(tok_path)
        tok.name = f"bpe-{args.vocab_size}"
    print(f"tokenizer {tok.name}: vocab {tok.vocab_size:,}, eos id {tok.eos_id} ({time.time() - t0:.1f}s)")

    sample = val_docs[:2000]
    chars = sum(len(d) for d in sample)
    toks = sum(len(ids) for ids in tok.encode_batch(sample))
    print(f"compression on 2,000 val docs: {chars / toks:.2f} chars/token")

    out = data_dir if args.tokenizer == "bpe" else data_dir.with_name(data_dir.name + "-gpt2")
    t0 = time.time()
    for split, docs in (("train", train_docs), ("val", val_docs)):
        meta = write_shards(docs, tok, out, split, args.shard_tokens)
        lens = np.load(out / f"{split}_doc_lens.npy")
        print(f"{split}: {meta[split]['tokens']:,} tokens in {len(meta[split]['shards'])} shard(s); "
              f"doc length mean {lens.mean():.0f}, median {np.median(lens):.0f}, max {lens.max()}")
    print(f"wrote {out} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
