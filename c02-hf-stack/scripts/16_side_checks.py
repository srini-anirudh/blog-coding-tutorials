"""Sections 4 and 9: two facts the other scripts do not print.

1. How long Datasets takes to hash a mapped function that reaches a tokenizer, with repeats, because timings vary.
2. The tile grid behind SmolVLM's image tokens, read from its row and column tokens.

Run: python c02-hf-stack/scripts/16_side_checks.py [--repeats 9]
"""

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.fingerprint import Hasher
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from hfstack.chat import assistant_labels_by_prefix
from hfstack.data import to_messages
from hfstack.utils import CHAT_LM, SMALL_VLM, be_polite, hardware_label

parser = argparse.ArgumentParser()
parser.add_argument("--repeats", type=int, default=9)
args = parser.parse_args()
be_polite()
print("timings below:", hardware_label())

tok = AutoTokenizer.from_pretrained(CHAT_LM)


def plain(row: dict) -> dict:
    """The control: a mapped function that reaches nothing large."""
    return row


def tokenize(row: dict) -> dict:
    """The same function as 05_datasets.py. It reaches the whole tokenizer through a global."""
    return assistant_labels_by_prefix(tok, to_messages(row)["messages"])


print(f"\n--- Hasher.hash, {args.repeats} repeats in one process, seconds ---")
print(f"{'object':<28} {'first':>7} {'min':>7} {'median':>7} {'max':>7}  distinct hashes")
for name, target in (("function reaching nothing", plain), ("tokenizer alone", tok), ("function reaching tokenizer", tokenize)):
    times, hashes = [], set()
    for _ in range(args.repeats):
        start = time.perf_counter()
        hashes.add(Hasher.hash(target))
        times.append(time.perf_counter() - start)
    print(f"{name:<28} {times[0]:>7.3f} {min(times):>7.3f} {statistics.median(times):>7.3f} {max(times):>7.3f}  {len(hashes)}")

print("\n--- SmolVLM tile grid, read from the row and column tokens ---")
processor = AutoProcessor.from_pretrained(SMALL_VLM)
print("image_processor.size:", processor.image_processor.size)
print("max_image_size      :", processor.image_processor.max_image_size)
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What does it say?"}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
for width, height in ((640, 480), (300, 300)):
    batch = processor(text=[prompt], images=[[Image.new("RGB", (width, height), (240, 240, 235))]], return_tensors="pt")
    tokens = processor.tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist())
    cells = {(int(r), int(c)) for r, c in (m.groups() for m in map(re.compile(r"<row_(\d+)_col_(\d+)>").match, tokens) if m)}
    rows, cols = max(r for r, _ in cells), max(c for _, c in cells)
    print(
        f"{width} x {height}: {rows} rows x {cols} columns = {rows * cols} tiles, "
        f"plus {tokens.count('<global-img>')} global = {batch['pixel_values'].shape[1]} in pixel_values"
    )
