"""Datasets throughput: per-row map vs batched map vs two processes, tokenizing arithmetic chats.

Run: python c02-hf-stack/benchmarks/bench_map.py [--rows 20000]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset
from transformers import AutoTokenizer

from hfstack.data import make_arithmetic_rows, question
from hfstack.utils import CHAT_LM, be_polite, hardware_label

be_polite()
parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=20_000)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(CHAT_LM)
data = Dataset.from_list(make_arithmetic_rows(args.rows))
data = data.map(lambda r: {"text": question(r) + " " + str(r["answer"])}, keep_in_memory=True)


def one(row: dict) -> dict:
    return tok(row["text"])


def many(batch: dict) -> dict:
    return tok(batch["text"])


def run(name: str, **kwargs) -> None:
    fn = many if kwargs.get("batched") else one
    start = time.perf_counter()
    data.map(fn, load_from_cache_file=False, keep_in_memory=True, **kwargs)
    seconds = time.perf_counter() - start
    print(f"{name:<34} {seconds:>7.2f} s {args.rows / seconds:>12,.0f} rows/s")


print(f"{args.rows:,} rows, {hardware_label()}, TOKENIZERS_PARALLELISM=false")
run("per row")
run("batched, batch_size=1000", batched=True)
run("batched, batch_size=1000, 2 procs", batched=True, num_proc=2)
