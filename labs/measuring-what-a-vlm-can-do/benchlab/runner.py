"""Batched inference for two small public VLMs and their language backbones, plus data loaders.

SmolVLM-256M-Instruct and SmolVLM-500M-Instruct are Idefics3 models. Their text decoders have the
shape of SmolLM2-135M and SmolLM2-360M, so the matching SmolLM2 Instruct checkpoints serve as the
"base LLM" in MMStar's leakage test. They are not the exact decoder weights inside each VLM:
the VLM decoders were trained further on multimodal data. That is the comparison MMStar makes
too (the VLM against the LLM it was built from).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import torch
from PIL import Image

from .mc import MCItem, parse_mmstar
from .report import command_line, hardware  # noqa: F401  (re-exported for the run script)

MODELS = {
    "256m": ("HuggingFaceTB/SmolVLM-256M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    "500m": ("HuggingFaceTB/SmolVLM-500M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"),
}


# --- Data ------------------------------------------------------------------------------------

@dataclass
class MMStarRow:
    index: int
    category: str
    image: Image.Image
    item: MCItem


def load_mmstar(limit: int | None = None) -> tuple[list[MMStarRow], int]:
    """MMStar's 1,500 questions from the Hub (Lin-Chen/MMStar). Returns rows and the number skipped."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    table = pq.read_table(hf_hub_download("Lin-Chen/MMStar", "mmstar.parquet", repo_type="dataset"))
    rows, skipped = [], 0
    for r in table.to_pylist()[:limit]:
        try:
            item = parse_mmstar(r["question"], r["answer"])
        except ValueError:
            skipped += 1
            continue
        img = Image.open(io.BytesIO(r["image"])).convert("RGB")
        rows.append(MMStarRow(r["index"], r["category"], img, item))
    return rows, skipped


POPE_SETTINGS = ("random", "popular", "adversarial")


def load_pope(setting: str, limit: int | None = None) -> list[dict]:
    """POPE (lmms-lab/POPE): 3,000 yes/no questions per setting over the same 500 COCO images."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("lmms-lab/POPE", f"Full/{setting}-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(path).to_pylist()[:limit]
    cache: dict[str, Image.Image] = {}
    out = []
    for r in rows:
        key = r["image_source"]
        if key not in cache:
            cache[key] = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        out.append({"question": r["question"], "answer": r["answer"].strip().lower(), "image": cache[key]})
    return out


# --- Models ----------------------------------------------------------------------------------

def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VLM:
    """SmolVLM with batched greedy generation. `split` toggles Idefics3 image tiling."""

    def __init__(self, model_id: str, device: torch.device, split: bool = True):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.proc = AutoProcessor.from_pretrained(model_id)
        self.proc.tokenizer.padding_side = "left"
        self.split = split
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.device = device
        self.eou = self.proc.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
        self.image_token = self.proc.tokenizer.convert_tokens_to_ids("<image>")

    def _prompt(self, text: str, with_image: bool) -> str:
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": text}]
        return self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)

    def _encode_batch(self, prompts: list[str], images: list[Image.Image] | None) -> dict:
        """Tokenise and (for tiled images) resize and split. This is the CPU-heavy part of a batch."""
        self.proc.image_processor.do_image_splitting = self.split
        if images is None:
            enc = self.proc(text=prompts, return_tensors="pt", padding=True)
        else:
            enc = self.proc(text=prompts, images=[[im] for im in images], return_tensors="pt", padding=True)
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(self.model.dtype)  # halve the bytes sent to the main process
        return dict(enc)

    @torch.no_grad()
    def generate(self, texts: list[str], images: list[Image.Image] | None, max_new_tokens: int = 24,
                 batch_size: int = 16, workers: int = 0) -> tuple[list[str], float]:
        """Greedy replies, and the mean number of image tokens per prompt (0 without images).

        Tiled preprocessing costs about 0.3 s per image on one core, far more than a forward pass on
        a GPU, so `workers` worker processes encode batches while the model generates.
        """
        starts = list(range(0, len(texts), batch_size))
        encode = lambda s: self._encode_batch(  # noqa: E731
            [self._prompt(t, images is not None) for t in texts[s:s + batch_size]],
            None if images is None else images[s:s + batch_size])
        loader = torch.utils.data.DataLoader(_Lazy(starts, encode), batch_size=None, num_workers=workers,
                                             prefetch_factor=2 if workers else None, worker_init_fn=_one_thread)
        replies, n_img_tokens = [], 0
        for enc in loader:
            n_img_tokens += int((enc["input_ids"] == self.image_token).sum())
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=self.eou, pad_token_id=self.proc.tokenizer.pad_token_id)
            replies += self.proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [r.strip() for r in replies], n_img_tokens / max(1, len(texts))


class _Lazy(torch.utils.data.Dataset):
    """Applies `fn` to each item on demand, so DataLoader workers do the encoding."""

    def __init__(self, items, fn):
        self.items, self.fn = items, fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.fn(self.items[i])


def _one_thread(_worker_id: int) -> None:
    torch.set_num_threads(1)


class LLM:
    """A text-only SmolLM2 Instruct model with the same batched greedy generation."""

    def __init__(self, model_id: str, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.device = device

    @torch.no_grad()
    def generate(self, texts: list[str], max_new_tokens: int = 24, batch_size: int = 32) -> list[str]:
        replies = []
        for s in range(0, len(texts), batch_size):
            prompts = [self.tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                                    add_generation_prompt=True) for t in texts[s:s + batch_size]]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            replies += self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [r.strip() for r in replies]
