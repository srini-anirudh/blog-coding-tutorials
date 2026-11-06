"""Three small text domains for the lab: prose, code and arithmetic.

- prose: Wikipedia articles from WikiText-2 (downloaded once from the Hugging Face Hub, about 13 MB).
- code: the `.py` files of this Python's standard library. Every Python install has them, so no download.
- math: synthetic arithmetic lines, generated from a seed. Unbounded, so it never runs out.

Each loader returns a list of documents (strings). `as_bytes` turns a list into one uint8 array,
with byte 0 between documents as the end-of-document token.
"""

from __future__ import annotations

import glob
import os
import random
import re
import sysconfig

import numpy as np

EOD = 0  # byte 0 never appears in the text we load, so it marks a document boundary


def _detok(line: str) -> str:
    """Undo the spaces WikiText puts around punctuation, so the text reads like normal prose."""
    line = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    line = re.sub(r" ([.,;:!?)'])", r"\1", line)
    line = re.sub(r"([(]) ", r"\1", line)
    return line.replace(" 's", "'s").strip()


def wikitext_docs(split: str = "train") -> list[str]:
    """WikiText-2 articles. An article starts at a line like ' = Title = ' (one '=' on each side)."""
    from datasets import load_dataset

    lines = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]
    docs, cur = [], []
    for line in lines:
        if re.fullmatch(r" = [^=].* = \n", line):
            if cur:
                docs.append("\n".join(cur))
            cur = [line.strip(" =\n")]
        elif line.strip():
            s = line.strip()
            if s.startswith("="):  # section heading such as ' = = Plot = = '
                s = s.strip("= ").strip()
            cur.append(_detok(s))
    if cur:
        docs.append("\n".join(cur))
    return docs


def stdlib_code_docs(split: str = "train") -> list[str]:
    """Top-level standard-library modules. Sorted by name; every 10th file is the validation split."""
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    picked = [f for i, f in enumerate(files) if (i % 10 == 9) == (split != "train")]
    out = []
    for f in picked:
        with open(f, encoding="utf-8", errors="replace") as fh:
            text = fh.read().replace("\x00", "")
        if text.strip():
            out.append(text)
    return out


def math_docs(n_docs: int, seed: int = 0, lines_per_doc: int = 20) -> list[str]:
    """Synthetic arithmetic, for example '417 + 38 = 455'. Deterministic for a given seed."""
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        lines = []
        for _ in range(lines_per_doc):
            a, b = rng.randint(0, 999), rng.randint(0, 999)
            op = rng.choice("+-*")
            c = {"+": a + b, "-": a - b, "*": a * b}[op]
            lines.append(f"{a} {op} {b} = {c}")
        docs.append("\n".join(lines))
    return docs


def as_bytes(docs: list[str]) -> np.ndarray:
    """Concatenate UTF-8 bytes of every document, with EOD after each one."""
    parts = []
    for d in docs:
        parts.append(np.frombuffer(d.encode("utf-8"), dtype=np.uint8))
        parts.append(np.array([EOD], dtype=np.uint8))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)


def load_domains(seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Byte streams for the three domains, train and validation."""
    return {
        "prose": {"train": as_bytes(wikitext_docs("train")), "val": as_bytes(wikitext_docs("validation"))},
        "code": {"train": as_bytes(stdlib_code_docs("train")), "val": as_bytes(stdlib_code_docs("val"))},
        "math": {"train": as_bytes(math_docs(20000, seed)), "val": as_bytes(math_docs(500, seed + 1))},
    }
