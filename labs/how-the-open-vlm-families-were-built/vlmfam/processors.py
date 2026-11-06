"""Run the real Hugging Face processors and count the tokens they spend on one image.

Only processor files are downloaded (tokenizer and preprocessing configs, a few MB each), never weights.
For each family we read the processor's own settings (pinpoints, pixel limits, longest edge) and pass them to the
from-scratch rule in `tokens.py`, so a mismatch means the rule is wrong, not the settings.

Some processors (Qwen2-VL, LLaVA-OneVision, InternVL, Pixtral) also load a video processor that needs torchvision.
This environment has no torchvision built for its torch, so for those four we run the PIL image processor and
then apply the processor's own counting code: `image_grid_thw` for Qwen, `num_patches` for InternVL, the resized
size for Pixtral, and `LlavaOnevisionProcessor._get_number_of_features` for OneVision. The other four run the full
processor and we count placeholder ids in `input_ids`.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from . import tokens as T

# (rule name, repo). Gated Google repos are read through the public unsloth mirror.
PROCESSORS = [
    ("LLaVA-1.5", "llava-hf/llava-1.5-7b-hf"),
    ("LLaVA-NeXT", "llava-hf/llava-v1.6-mistral-7b-hf"),
    ("LLaVA-OneVision", "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"),
    ("Qwen2/2.5-VL", "Qwen/Qwen2-VL-2B-Instruct"),
    ("Qwen2/2.5-VL", "Qwen/Qwen2.5-VL-3B-Instruct"),
    ("InternVL 1.5-3", "OpenGVLab/InternVL3-1B-hf"),
    ("Idefics3", "HuggingFaceM4/Idefics3-8B-Llama3"),
    ("Idefics3", "HuggingFaceTB/SmolVLM-256M-Instruct"),
    ("Pixtral", "mistral-community/pixtral-12b"),
    ("Gemma 3", "unsloth/gemma-3-4b-it"),
]
IMAGE_ONLY = {"LLaVA-OneVision", "Qwen2/2.5-VL", "InternVL 1.5-3", "Pixtral"}

SIZES = [(w, h) for _, w, h in T.PRESETS] + [(640, 480), (500, 1500), (1234, 567), (2000, 600), (224, 224),
                                               (100, 800)]


def image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(width * 10007 + height)
    return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


def processor_config(repo: str) -> dict:
    from huggingface_hub import hf_hub_download

    try:
        return json.load(open(hf_hub_download(repo, "processor_config.json")))
    except Exception:
        return {}


class Probe:
    """One repo's processor (or image processor), its settings, and a `measure(width, height)` method."""

    def __init__(self, label: str, repo: str):
        from transformers import AutoImageProcessor, AutoProcessor

        self.label, self.repo = label, repo
        self.cfg = processor_config(repo)
        if label in IMAGE_ONLY:
            self.proc = None
            self.ip = AutoImageProcessor.from_pretrained(repo, backend="pil")
        else:
            self.proc = AutoProcessor.from_pretrained(repo)
            self.ip = self.proc.image_processor
        self.how = "image processor + its counting code" if self.proc is None else "full processor, ids counted"

    # -- what the processor does
    def measure(self, width: int, height: int) -> int:
        img = image(width, height)
        if self.label == "Qwen2/2.5-VL":
            t, h, w = self.ip(images=[img], return_tensors="np")["image_grid_thw"][0]
            return int(t * h * w) // self.ip.merge_size ** 2
        if self.label == "InternVL 1.5-3":
            n = self.ip(images=[img], crop_to_patches=True, return_tensors="np")["num_patches"][0]
            return int(n) * self.cfg.get("image_seq_length", 256)
        if self.label == "Pixtral":
            h, w = self.ip(images=[img], return_tensors="np")["image_sizes"][0]
            p = self.cfg.get("patch_size", 16) * self.cfg.get("spatial_merge_size", 1)
            return int(h // p) * int(w // p + 1)  # one [IMG] per patch, a break or end token per row
        if self.label == "LLaVA-OneVision":
            from transformers import LlavaOnevisionProcessor

            p = LlavaOnevisionProcessor.__new__(LlavaOnevisionProcessor)
            p.image_processor = self.ip
            p.num_image_tokens = self.cfg["num_image_tokens"]
            p.vision_aspect_ratio = self.cfg.get("vision_aspect_ratio", "anyres_max_9")
            size = self.ip.size
            n = p._get_number_of_features(height, width, size["height"], size["width"])
            return n - (1 if self.cfg.get("vision_feature_select_strategy") == "default" else 0)
        return self._count_ids(img)

    def _count_ids(self, img) -> int:
        proc, tok = self.proc, self.proc.tokenizer
        name = type(proc).__name__
        if name.startswith("Gemma3"):
            marker, counted = proc.boi_token, ["<image_soft_token>"]
        else:
            marker, counted = "<image>", [getattr(proc, "image_token", "<image>")]
        ids = set(tok.convert_tokens_to_ids(counted))
        out = proc(text=[f"{marker}\nDescribe."], images=[[img]], return_tensors="np")
        return sum(1 for i in out["input_ids"][0].tolist() if i in ids)

    # -- the from-scratch rule with this processor's settings
    def rule(self, width: int, height: int) -> int:
        ip, label = self.ip, self.label
        if label == "LLaVA-NeXT":
            pins = [(w, h) for h, w in ip.image_grid_pinpoints]  # HF stores (height, width)
            return T.llava_next(width, height, pinpoints=pins).tokens
        if label == "Qwen2/2.5-VL":
            return T.qwen2_vl(width, height, patch=ip.patch_size, merge=ip.merge_size,
                              min_pixels=getattr(ip, "min_pixels", None) or ip.size["shortest_edge"],
                              max_pixels=getattr(ip, "max_pixels", None) or ip.size["longest_edge"]).tokens
        if label == "InternVL 1.5-3":
            return T.internvl(width, height, tile=ip.size["height"],
                              tokens_per_tile=self.cfg.get("image_seq_length", 256), max_tiles=ip.max_patches).tokens
        if label == "Idefics3":
            return T.idefics3(width, height, longest_edge=ip.size["longest_edge"],
                              tile=ip.max_image_size["longest_edge"], tokens_per_tile=self.proc.image_seq_len).tokens
        if label == "Pixtral":
            return T.pixtral(width, height, patch=self.cfg.get("patch_size", 16),
                             longest_edge=ip.size["longest_edge"]).tokens
        return T.RULES[label](width, height).tokens

    def settings(self) -> str:
        ip = self.ip
        keys = ["size", "max_image_size", "min_pixels", "max_pixels", "max_patches", "patch_size", "merge_size",
                "do_pan_and_scan"]
        found = {}
        for k in keys:
            v = getattr(ip, k, None)
            if v is None:
                continue
            if hasattr(v, "__dataclass_fields__") or hasattr(v, "items"):
                items = v.items() if hasattr(v, "items") else vars(v).items()
                v = {a: b for a, b in items if b is not None}
            found[k] = v
        if getattr(ip, "image_grid_pinpoints", None):
            found["pinpoints"] = len(ip.image_grid_pinpoints)
        for k in ("num_image_tokens", "image_seq_length", "image_seq_len"):
            v = self.cfg.get(k, getattr(self.proc, k, None) if self.proc is not None else None)
            if v is not None:
                found[k] = v
        return ", ".join(f"{k}={v}" for k, v in found.items())
