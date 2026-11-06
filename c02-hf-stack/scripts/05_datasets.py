"""Sections 8-10: Dataset, DatasetDict, Arrow, map/filter/shuffle/select, fingerprints, streaming.

Run: python c02-hf-stack/scripts/05_datasets.py
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from transformers import AutoTokenizer

from hfstack.chat import assistant_labels_by_prefix
from hfstack.data import arithmetic_dataset_dict, make_arithmetic_rows, to_messages, write_jsonl
from hfstack.utils import CHAT_LM, be_polite

be_polite()
tmp = Path(tempfile.mkdtemp())

# 1. load_dataset works on local files too; the Hub is one source among several.
path = write_jsonl(make_arithmetic_rows(2_000), tmp / "arithmetic.jsonl")
data = load_dataset("json", data_files={"train": str(path)}, cache_dir=str(tmp / "cache"))
print(type(data).__name__, "->", data)
train = data["train"]
print("features:", train.features)

# 2. Arrow: columns on disk, memory-mapped. Indexing reads and decodes one row.
print("\nrow 0        :", train[0])
print("cache files  :", [Path(f["filename"]).name for f in train.cache_files])
print("arrow table  :", type(train.data).__name__, "| nbytes:", train.data.nbytes)
python_list = train.to_list()
print("python list  :", sys.getsizeof(python_list) + sum(sys.getsizeof(r) for r in python_list), "bytes (shallow, no int objects)")

# 3. Transformations return new datasets. The fingerprint names each state.
print("\nfingerprint            :", train._fingerprint)
positive = train.filter(lambda row: row["answer"] >= 0)
print("filter(answer >= 0)    :", positive._fingerprint, len(positive), "rows")
shuffled = train.shuffle(seed=0)
print("shuffle(seed=0)        :", shuffled._fingerprint, "| indices mapping:", shuffled._indices is not None)
print("select(range(4))       :", train.select(range(4))._fingerprint, "| ids", train.select(range(4))["id"][:4])
print("first ids after shuffle:", shuffled["id"][:6])

# 4. A cached map: same function and same input fingerprint -> read from disk, no recompute.
tok = AutoTokenizer.from_pretrained(CHAT_LM)


def tokenize(row: dict) -> dict:
    return assistant_labels_by_prefix(tok, to_messages(row)["messages"])


for attempt in ("first ", "second"):
    start = time.perf_counter()
    tokenized = train.map(tokenize, remove_columns=train.column_names)
    print(f"\n{attempt} map: {time.perf_counter() - start:6.2f} s | fingerprint {tokenized._fingerprint}")
print("columns after remove_columns:", tokenized.column_names)
print("row 0 lengths:", {k: len(v) for k, v in tokenized[0].items()})

# 5. Batched map changes what the function receives: a dict of lists, and it may change the row count.
def explode(batch: dict) -> dict:
    return {"id": [i for i in batch["id"] for _ in range(2)], "copy": [c for _ in batch["id"] for c in (0, 1)]}


doubled = train.select(range(4)).map(explode, batched=True, batch_size=2, remove_columns=train.column_names)
print("\nbatched map, 4 rows in ->", len(doubled), "rows out:", doubled[:4])

# 6. Streaming: an IterableDataset has no length and no random access, only an order.
stream = load_dataset("json", data_files=str(path), split="train", streaming=True)
print("\n", type(stream).__name__, "| has __len__:", hasattr(stream, "__len__"))
print("first row      :", next(iter(stream)))
transformed = stream.filter(lambda r: r["answer"] >= 0).map(lambda r: {"text": f"{r['a']}{r['op']}{r['b']}"})
print("lazy transforms:", next(iter(transformed)))
buffered = stream.shuffle(seed=0, buffer_size=100)
print("buffer shuffle, first ids:", [row["id"] for row, _ in zip(buffered, range(6))])
print("take(2)        :", [row["id"] for row in stream.take(2)])
print("skip(1998) ids :", [row["id"] for row in stream.skip(1998)])
