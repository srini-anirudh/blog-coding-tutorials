"""Unified sample -> model tensors: chat template, image-token expansion, loss mask, padding.

For one VQA sample with a single 224 px image (1 tile, 49 visual tokens) the sequence is

    <|im_start|>user\n  <image> x 49  How many red circles are there? ...<|im_end|>\n
    <|im_start|>assistant\n  2<|im_end|>\n

and the labels are -100 everywhere except the answer tokens and the closing <|im_end|>, which is
what teaches the model to stop. Visual tokens are never supervised: there is nothing to predict there.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .resolution import preprocess
from .tokenizer import IM_END, IM_START, IMAGE_TOKEN

IGNORE = -100


@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]
    pixel_values: torch.Tensor | None  # (n_tiles, 3, H, W) for all images of the sample, in order
    tiles_per_image: list[int]
    task: str
    meta: dict


class TinyVLMProcessor:
    def __init__(self, tokenizer, tokens_per_tile: int = 49, tile: int = 224, strategy: str = "tile",
                 max_tiles: int = 4, max_length: int = 1024):
        self.tok = tokenizer
        self.tokens_per_tile = tokens_per_tile
        self.tile, self.strategy, self.max_tiles = tile, strategy, max_tiles
        self.max_length = max_length
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.eos_id = tokenizer.convert_tokens_to_ids(IM_END)
        self.pad_id = tokenizer.pad_token_id

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def encode(self, sample: dict, for_generation: bool = False) -> Encoded:
        """One sample -> token ids, labels and pixel tiles. Generation drops the final answer."""
        messages = sample["messages"]
        if for_generation:
            messages = messages[:-1]
        pix = [preprocess(img, self.strategy, self.tile, self.max_tiles) for img in sample["images"]]
        tiles = [p.shape[0] for p in pix]
        image_iter = iter(tiles)
        ids: list[int] = []
        labels: list[int] = []

        def add(tokens: list[int], supervised: bool) -> None:
            ids.extend(tokens)
            labels.extend(tokens if supervised else [IGNORE] * len(tokens))

        for m in messages:
            if m["role"] == "text":  # plain-text replay: no template, every token is a target
                add(self._ids(m["content"][0]["text"]) + [self.eos_id], supervised=True)
                continue
            is_answer = m["role"] == "assistant"
            add(self._ids(f"{IM_START}{m['role']}\n"), supervised=False)
            for c in m["content"]:
                if c["type"] == "image":
                    add([self.image_token_id] * (next(image_iter) * self.tokens_per_tile), supervised=False)
                else:
                    add(self._ids(c["text"]), supervised=is_answer)
            add([self.eos_id], supervised=is_answer)
            add(self._ids("\n"), supervised=False)
        if for_generation:
            add(self._ids(f"{IM_START}assistant\n"), supervised=False)
        if len(ids) > self.max_length:
            # Truncating would cut visual tokens loose from their tiles, so refuse instead.
            raise ValueError(f"sample has {len(ids)} tokens > max_length {self.max_length}")
        return Encoded(ids, labels, torch.cat(pix) if pix else None, tiles, sample["task"], sample.get("meta", {}))

    def collate(self, items: list[Encoded], padding_side: str = "right") -> dict:
        """Pad to the longest sequence. Left padding for generation, so every prompt ends at the same column."""
        n = max(len(e.input_ids) for e in items)
        input_ids = torch.full((len(items), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(items), n), IGNORE, dtype=torch.long)
        attention_mask = torch.zeros((len(items), n), dtype=torch.long)
        for b, e in enumerate(items):
            L = len(e.input_ids)
            sl = slice(0, L) if padding_side == "right" else slice(n - L, n)
            input_ids[b, sl] = torch.tensor(e.input_ids)
            labels[b, sl] = torch.tensor(e.labels)
            attention_mask[b, sl] = 1
        pix = [e.pixel_values for e in items if e.pixel_values is not None]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat(pix) if pix else None,
            "tasks": [e.task for e in items],
            "meta": [e.meta for e in items],
        }

    def __call__(self, samples: list[dict], for_generation: bool = False) -> dict:
        enc = [self.encode(s, for_generation) for s in samples]
        return self.collate(enc, padding_side="left" if for_generation else "right")
