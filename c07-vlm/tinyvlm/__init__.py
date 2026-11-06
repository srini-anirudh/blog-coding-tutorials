"""tinyvlm: a vision-language model assembled from a pretrained SigLIP encoder and SmolLM2.

Public contract (C8 "Train a Real VLM" builds on it):
    VLMConfig, TinyVLM, TinyVLMProcessor, TinyVLMImageProcessor, build_tinyvlm
    TinyVLM.forward(input_ids, pixel_values, attention_mask=None, labels=None) -> .loss, .logits
    TinyVLM.generate(input_ids, pixel_values, attention_mask, **hf_generate_kwargs) -> new token ids
    TinyVLM submodules: .vision, .projector, .lm
"""

from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText

from .build import LM_ID, VISION_ID, build_tinyvlm
from .chat import IMAGE_TOKEN, apply_chat_template, chat_segments
from .configuration import VLMConfig
from .device import describe_device, pick_device
from .image_processing import TinyVLMImageProcessor
from .modeling import TinyVLM
from .processing import IGNORE_INDEX, TinyVLMProcessor

# Registering lets AutoConfig / AutoModel load a saved TinyVLM directory like any Hub model.
AutoConfig.register("tinyvlm", VLMConfig, exist_ok=True)
AutoModel.register(VLMConfig, TinyVLM, exist_ok=True)
AutoModelForImageTextToText.register(VLMConfig, TinyVLM, exist_ok=True)

__all__ = [
    "IGNORE_INDEX",
    "IMAGE_TOKEN",
    "LM_ID",
    "VISION_ID",
    "TinyVLM",
    "TinyVLMImageProcessor",
    "TinyVLMProcessor",
    "VLMConfig",
    "apply_chat_template",
    "build_tinyvlm",
    "chat_segments",
    "describe_device",
    "pick_device",
]
