"""One sample format for every task: {"images": [...], "messages": [...]}.

Captions, VQA, OCR, documents, charts, grounding and plain text all become the same dict, so the
processor, the mixture sampler and the trainer never branch on the task:

    {
        "task": "vqa",
        "images": [<PIL.Image 224x224>],
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How many red circles are there?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "2"}]},
        ],
        "meta": {"answer": "2"},          # whatever the metric for this task needs
    }

A text-only replay sample has no images and a single {"role": "text"} message: plain text that is
supervised end to end, with no chat wrapper, exactly like language-model pretraining data.
"""

from __future__ import annotations

import random
import zlib
from typing import Callable

from torch.utils.data import Dataset

from . import grounding as G
from .synth import charts, shapes, text

TASKS = ("caption", "vqa", "ocr", "doc", "chart", "grounding")

# Nominal corpus sizes. They mimic the imbalance of real mixtures (captions are plentiful,
# documents and charts are scarce), and they are what "proportional" mixing is proportional to.
TASK_SIZES = {"caption": 40_000, "vqa": 20_000, "ocr": 8_000, "doc": 1_000, "chart": 2_000, "grounding": 4_000}


def user(text_: str, n_images: int = 1) -> dict:
    return {"role": "user", "content": [{"type": "image"}] * n_images + [{"type": "text", "text": text_}]}


def assistant(text_: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text_}]}


def sample(task: str, images: list, question: str, answer: str, **meta) -> dict:
    return {"task": task, "images": images, "messages": [user(question, len(images)), assistant(answer)],
            "meta": {"answer": answer, **meta}}


def validate(s: dict) -> None:
    """Fail loudly on the two mistakes that silently ruin VLM training data."""
    n_slots = sum(c.get("type") == "image" for m in s["messages"] for c in _content(m))
    if n_slots != len(s["images"]):
        raise ValueError(f"{n_slots} image slots in messages but {len(s['images'])} images")
    roles = [m["role"] for m in s["messages"]]
    if roles != ["text"] and (not roles or roles[-1] != "assistant"):
        raise ValueError(f"a training sample must end with an assistant turn, got roles {roles}")


def _content(m: dict) -> list[dict]:
    return m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]


def rng_for(task: str, split: str, index: int) -> random.Random:
    """Sample `index` of a task is a pure function of (task, split, index): reproducible and resumable."""
    return random.Random(zlib.crc32(f"{task}/{split}/{index}".encode()))


# ---- one generator per task ------------------------------------------------------------------

def gen_caption(rng: random.Random, **_) -> dict:
    img, objs = shapes.make_scene(rng, 1, 4)
    return sample("caption", [img], rng.choice(["Describe the image.", "What is in this picture?"]),
                  shapes.caption(objs))


def gen_vqa(rng: random.Random, **_) -> dict:
    img, objs = shapes.make_scene(rng, 1, 5)
    q, a = shapes.vqa_pair(rng, objs)
    return sample("vqa", [img], q + " Answer with one word.", a)


def gen_ocr(rng: random.Random, **_) -> dict:
    img, words = text.make_ocr(rng)
    return sample("ocr", [img], "What does the text in the image say?", words)


def gen_doc(rng: random.Random, **_) -> dict:
    img, fields = text.make_document(rng)
    key = rng.choice(list(fields))
    return sample("doc", [img], text.DOC_QUESTIONS[key], fields[key], field=key)


def gen_chart(rng: random.Random, **_) -> dict:
    img, values = charts.make_chart(rng)
    q, a = charts.chart_question(rng, values)
    return sample("chart", [img], q, a, values=values)


def gen_grounding(rng: random.Random, fmt: str = "tokens", **_) -> dict:
    """Boxes 60% of the time, points 20%, mask outlines 20%. Targets are exact because we drew the shapes."""
    img, objs = shapes.make_scene(rng, 2, 4, distinct=True)
    o = rng.choice(objs)
    w, h = img.size
    r = rng.random()
    if r < 0.6:
        return sample("grounding", [img], f"Find the {o.color} {o.shape}. Give its bounding box.",
                      G.encode_box(o.box, w, h, fmt), kind="box", box=o.box, size=(w, h), fmt=fmt)
    if r < 0.8:
        return sample("grounding", [img], f"Point to the {o.color} {o.shape}.",
                      G.encode_point(o.center, w, h, fmt), kind="point", box=o.box, size=(w, h), fmt=fmt)
    return sample("grounding", [img], f"Trace the outline of the {o.color} {o.shape}.",
                  G.encode_polygon(o.polygon(), w, h, fmt), kind="mask", box=o.box, size=(w, h), fmt=fmt,
                  rle=G.mask_to_rle(shapes.object_mask(o, w)))


GENERATORS: dict[str, Callable[..., dict]] = {
    "caption": gen_caption, "vqa": gen_vqa, "ocr": gen_ocr,
    "doc": gen_doc, "chart": gen_chart, "grounding": gen_grounding,
}


def make_example(task: str, index: int, split: str = "train", grounding_format: str = "tokens") -> dict:
    s = GENERATORS[task](rng_for(task, split, index), fmt=grounding_format)
    validate(s)
    return s


def text_sample(passage: str) -> dict:
    return {"task": "text", "images": [], "messages": [{"role": "text", "content": [{"type": "text", "text": passage}]}],
            "meta": {}}


class TaskDataset(Dataset):
    """A finite synthetic dataset: `size` reproducible samples of one task."""

    def __init__(self, task: str, size: int, split: str = "train", grounding_format: str = "tokens"):
        self.task, self.size, self.split, self.fmt = task, size, split, grounding_format

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> dict:
        return make_example(self.task, i % self.size, self.split, self.fmt)
