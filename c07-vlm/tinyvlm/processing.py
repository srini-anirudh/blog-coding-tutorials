"""TinyVLMProcessor = tokenizer + image processor + the rule that ties them together.

The rule: every <image> marker in the text becomes num_crops x image_seq_len placeholder tokens,
because that is exactly how many feature vectors the model will produce for that picture.

Batch produced for the model (B samples, padded length N, n crops in total):
    input_ids       B x N      long
    attention_mask  B x N      long, 1 = real token, 0 = padding
    labels          B x N      long, -100 where no loss is taken (training only)
    pixel_values    n x 3 x 224 x 224
    num_crops       list[int], crops per picture, in placeholder order
    image_index     B x N      long, -1 for text, k for tokens of the k-th picture of that sample
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import torch
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .chat import IMAGE_TOKEN, apply_chat_template, chat_segments
from .image_processing import TinyVLMImageProcessor

IGNORE_INDEX = -100
PROCESSOR_CONFIG = "processor_config.json"


class TinyVLMProcessor:
    def __init__(
        self,
        image_processor: TinyVLMImageProcessor,
        tokenizer: PreTrainedTokenizerBase,
        image_seq_len: int,
        image_token: str = IMAGE_TOKEN,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_seq_len = image_seq_len
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        if self.image_token_id == tokenizer.unk_token_id or self.image_token_id is None:
            raise ValueError(f"{image_token!r} is not in the tokenizer; add it as a special token first")

    # ---- text side -----------------------------------------------------------------------------
    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = False) -> str:
        return apply_chat_template(messages, add_generation_prompt)

    def tokens_for_image(self, num_crops: int) -> int:
        return num_crops * self.image_seq_len

    def _expand(self, text: str, crops: Iterator[int]) -> str:
        """Replace each <image> marker by the right number of placeholders (consumes `crops`)."""
        pieces = text.split(self.image_token)
        out = [pieces[0]]
        for piece in pieces[1:]:
            num_crops = next(crops, None)
            if num_crops is None:
                raise ValueError("more <image> markers than pictures in the messages")
            out.append(self.image_token * self.tokens_for_image(num_crops))
            out.append(piece)
        return "".join(out)

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _image_index(self, ids: list[int], num_crops: list[int]) -> list[int]:
        """Boundary tracking: label each placeholder with the picture it belongs to."""
        owners = [k for k, c in enumerate(num_crops) for _ in range(self.tokens_for_image(c))]
        slots = [t for t, i in enumerate(ids) if i == self.image_token_id]
        if len(slots) != len(owners):
            raise ValueError(f"{len(slots)} <image> placeholders, but the pictures need {len(owners)}")
        index = [-1] * len(ids)
        for t, k in zip(slots, owners):
            index[t] = k
        return index

    # ---- inference path: already-rendered prompts ------------------------------------------------
    def __call__(
        self,
        text: str | list[str],
        images: list[Image.Image] | list[list[Image.Image]] | None = None,
        padding_side: str = "left",
    ) -> dict:
        """Tokenise prompts that contain one <image> marker per picture. Left padding suits generation."""
        texts = [text] if isinstance(text, str) else list(text)
        if images is None:
            per_sample = [[] for _ in texts]
        elif isinstance(text, str):
            per_sample = [list(images)]
        else:
            per_sample = [list(group) for group in images]
        examples = []
        for prompt, pics in zip(texts, per_sample, strict=True):
            if prompt.count(self.image_token) != len(pics):
                raise ValueError(f"prompt has {prompt.count(self.image_token)} <image> markers, {len(pics)} pictures")
            vision = self.image_processor(pics)
            ids = self._ids(self._expand(prompt, iter(vision["num_crops"])))
            examples.append({"input_ids": ids, "labels": None, **vision})
        return self.collate(examples, padding_side=padding_side)

    # ---- training path: messages with supervised spans -------------------------------------------
    def build_example(self, messages: list[dict], images: list[Image.Image] | None = None) -> dict:
        """One training example. Labels copy the ids of supervised pieces and are -100 elsewhere."""
        vision = self.image_processor(images or [])
        crops = iter(vision["num_crops"])
        input_ids: list[int] = []
        labels: list[int] = []
        for piece, supervised in chat_segments(messages):
            ids = self._ids(self._expand(piece, crops))
            input_ids += ids
            labels += [i if supervised and i != self.image_token_id else IGNORE_INDEX for i in ids]
        if next(crops, None) is not None:
            raise ValueError("more pictures than <image> markers in the messages")
        return {"input_ids": input_ids, "labels": labels, **vision}

    def collate(self, examples: list[dict], padding_side: str = "right") -> dict:
        """Pad a list of examples into one batch; padding gets attention 0 and label -100."""
        pad_id = self.tokenizer.pad_token_id
        length = max(len(e["input_ids"]) for e in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "image_index": []}
        for e in examples:
            ids, n_pad = list(e["input_ids"]), length - len(e["input_ids"])
            labels = e["labels"] if e["labels"] is not None else [IGNORE_INDEX] * len(ids)
            index = self._image_index(ids, e["num_crops"])
            parts = [
                (ids, [pad_id] * n_pad),
                ([1] * len(ids), [0] * n_pad),
                (labels, [IGNORE_INDEX] * n_pad),
                (index, [-1] * n_pad),
            ]
            for key, (real, pad) in zip(batch, parts):
                batch[key].append(pad + real if padding_side == "left" else real + pad)
        out = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        if all(e["labels"] is None for e in examples):
            del out["labels"]
        out["pixel_values"] = torch.cat([e["pixel_values"] for e in examples])
        out["num_crops"] = [c for e in examples for c in e["num_crops"]]
        return out

    # ---- decoding and serialisation ------------------------------------------------------------
    def batch_decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, directory: str | os.PathLike) -> None:
        """Same layout as a Hugging Face processor: tokenizer files, preprocessor_config.json, processor_config.json."""
        os.makedirs(directory, exist_ok=True)
        self.tokenizer.save_pretrained(directory)
        self.image_processor.save_pretrained(directory)
        config = {"processor_class": type(self).__name__, "image_token": self.image_token, "image_seq_len": self.image_seq_len}
        with open(os.path.join(directory, PROCESSOR_CONFIG), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, directory: str | os.PathLike) -> TinyVLMProcessor:
        with open(os.path.join(directory, PROCESSOR_CONFIG)) as f:
            config = json.load(f)
        return cls(
            image_processor=TinyVLMImageProcessor.from_pretrained(directory),
            tokenizer=AutoTokenizer.from_pretrained(directory),
            image_seq_len=config["image_seq_len"],
            image_token=config["image_token"],
        )
