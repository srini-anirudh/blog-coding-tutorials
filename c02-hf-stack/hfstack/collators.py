"""Data collators: turn a list of variable-length examples into one padded batch of tensors.

A `Dataset` yields one example at a time. A collator receives `batch_size` of them and must:
pad to a common length, build the attention mask, and set every label that should not be
trained on to -100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

IGNORE_INDEX = -100


@dataclass
class CausalLMCollator:
    """Right-pad `input_ids`, build `attention_mask`, and pad `labels` with -100.

    Input: a list of B dicts with `input_ids` (length N_i) and optionally `labels` (length N_i).
    Output: `input_ids`, `attention_mask`, `labels`, each a LongTensor of shape (B, N) with N = max N_i
    (rounded up to `pad_to_multiple_of` if set).

    Labels come from the example when present (so prompt masking survives). Otherwise they are a copy of
    `input_ids`. Padding is masked by position, never by token value: a pad token that equals EOS must
    still be trained on where it is a real EOS.
    """

    pad_token_id: int
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        longest = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            longest = (longest + m - 1) // m * m

        batch = len(features)
        input_ids = torch.full((batch, longest), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, longest), dtype=torch.long)
        labels = torch.full((batch, longest), IGNORE_INDEX, dtype=torch.long)

        for i, f in enumerate(features):
            ids = torch.as_tensor(f["input_ids"], dtype=torch.long)
            n = ids.numel()
            input_ids[i, :n] = ids
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.as_tensor(f["labels"], dtype=torch.long) if "labels" in f else ids

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class VisionChatCollator:
    """Collate image + conversation examples for an Idefics3-style VLM (SmolVLM).

    Each feature is {"image": PIL.Image, "messages": [user turn with an image, assistant turn]}.
    The processor expands the single `<image>` placeholder into many image tokens, so the prompt
    length depends on the image. Labels are -100 on padding, on image tokens and on the prompt.
    """

    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        tok = self.processor.tokenizer
        tok.padding_side = "right"  # training pads on the right; generation pads on the left
        images = [[f["image"]] for f in features]
        full_text = [self.processor.apply_chat_template(f["messages"]) for f in features]
        prompt_text = [
            self.processor.apply_chat_template(f["messages"][:-1], add_generation_prompt=True) for f in features
        ]

        batch = self.processor(text=full_text, images=images, padding=True, return_tensors="pt")
        prompts = self.processor(text=prompt_text, images=images, padding=True, return_tensors="pt")

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = IGNORE_INDEX
        image_token_id = tok.convert_tokens_to_ids("<image>")
        labels[batch["input_ids"] == image_token_id] = IGNORE_INDEX
        for i, prompt_len in enumerate(prompts["attention_mask"].sum(dim=1).tolist()):
            labels[i, :prompt_len] = IGNORE_INDEX
        batch["labels"] = labels
        return batch
