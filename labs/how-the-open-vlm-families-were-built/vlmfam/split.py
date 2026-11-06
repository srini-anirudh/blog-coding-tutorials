"""Where the parameters are: vision encoder, connector, language model, measured from real checkpoints.

`tensor_sizes(repo)` reads only the safetensors headers on the Hub (a few KB per shard, fetched with HTTP range
requests). The weights are never downloaded. `classify(name)` sorts each tensor into one part by its name, and
`split(sizes)` adds them up.

Cross-attention designs (Llama 3.2 Vision) hide their adapter inside the language model's layer list. Any layer
that holds a cross-attention tensor is counted as "adapter", since it is new, trained for vision, and absent from
the text-only model.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

PARTS = ("vision", "connector", "adapter", "language")

# Order matters: the first pattern that matches wins.
CONNECTOR = re.compile(
    r"(multi_modal_projector|visual\.merger|visual\.deepstack_merger|(^|\.)merger\.|(^|\.)mlp1\.|(^|\.)connector\."
    r"|(^|\.)projector\.|(^|\.)resampler\.|image_projector|image_pooling|image_newline|view_seperator"
    r"|(^|\.)aligner\.|mm_projector|vision_projection|vte\.|visual_tokenizer\.head)"
)
VISION = re.compile(
    r"((^|\.)vision_tower|(^|\.)vision_model|(^|\.)visual\.|(^|\.)vpm\.|^vision\.|(^|\.)image_vit|vision_backbone"
    r"|(^|\.)vit\.|visual_tokenizer|(^|\.)sam_model|vision_encoder|(^|\.)vision\.)"
)
LAYER = re.compile(r"(language_model|text_model|llm|model)\.(?:model\.)?layers\.(\d+)\.")


def numel(shape) -> int:
    return math.prod(shape) if shape else 1


def tensor_sizes(repo: str) -> dict[str, int]:
    """{tensor name: number of elements} from the safetensors headers of a Hub repo."""
    from huggingface_hub import HfApi

    meta = HfApi().get_safetensors_metadata(repo)
    out = {}
    for fm in meta.files_metadata.values():
        for name, t in fm.tensors.items():
            out[name] = numel(t.shape)
    return out


def classify(name: str) -> str:
    if CONNECTOR.search(name):
        return "connector"
    if VISION.search(name):
        return "vision"
    return "language"


def cross_attention_layers(names) -> set[str]:
    """Layer prefixes (for example 'language_model.model.layers.3.') that contain cross-attention tensors."""
    found = set()
    for n in names:
        if "cross_attn" in n:
            m = LAYER.search(n)
            if m:
                found.add(n[: m.end()])
    return found


def split(sizes: dict[str, int]) -> dict[str, int]:
    """Parameters per part. Packed or quantized tensors are not expected here; all repos used are bf16/fp16/fp32."""
    xattn = cross_attention_layers(sizes)
    parts = defaultdict(int)
    for name, n in sizes.items():
        part = classify(name)
        if part == "language" and any(name.startswith(p) for p in xattn):
            part = "adapter"
        parts[part] += n
    return {p: parts.get(p, 0) for p in PARTS}


def vision_share(parts: dict[str, int]) -> float:
    return parts["vision"] / sum(parts.values())
