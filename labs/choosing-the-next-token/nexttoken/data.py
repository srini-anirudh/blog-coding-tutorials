"""Prompts and reference text from WikiText-2 (test split), and a small report writer for the scripts."""

from __future__ import annotations

import re
from pathlib import Path

from .metrics import result_header

RESULTS = Path(__file__).resolve().parents[1] / "results"


def detok(text: str) -> str:
    """Undo WikiText's spacing: ' @-@ ' -> '-', ' , ' -> ', ' and so on."""
    text = re.sub(r" @(.)@ ", r"\1", text)
    text = re.sub(r" ([,.;:!?)'])", r"\1", text)
    text = re.sub(r"\( ", "(", text)
    return text.replace(" 's", "'s").strip()


def wiki_paragraphs(min_chars: int = 600) -> list[str]:
    from datasets import load_dataset

    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    return [detok(r) for r in rows if len(r) >= min_chars and not r.strip().startswith("=")]


def prompt_and_reference(tok, n: int, prompt_tokens: int = 32, ref_tokens: int = 100):
    """n (prompt ids (1, P), reference continuation ids (K,)) pairs, each from a different paragraph."""
    import torch

    out = []
    for para in wiki_paragraphs():
        ids = tok(para).input_ids
        if len(ids) >= prompt_tokens + ref_tokens:
            out.append((torch.tensor([ids[:prompt_tokens]]), torch.tensor(ids[prompt_tokens:prompt_tokens + ref_tokens])))
        if len(out) == n:
            break
    return out


class Report:
    """Print lines and save them to results/<name>.txt with the hardware line and command on top."""

    def __init__(self, script: str, name: str):
        self.path = RESULTS / f"{name}.txt"
        self.lines = [result_header(script), ""]
        print(self.lines[0])

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
        print(f"\nwrote {self.path.name}")
