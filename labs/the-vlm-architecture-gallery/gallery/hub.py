"""Check the gallery's token counts against real model configs on the Hugging Face Hub.

Only `config.json` / `preprocessor_config.json` are downloaded (a few kB each). For each model we
compute visual tokens per image (or per tile) from the config fields alone, and compare with the
number the post's fact sheet gives. We also read the language model's attention shape and compute
the KV cache per visual token, which the post's fifth explainer computes for one generic shape.

Only the derived fields are saved. Raw configs are not copied into the repo.
"""

from __future__ import annotations

import json

from .cost import kv_bytes_per_token
from .inputs import llava_next_tokens

# (repo, post's model id, what the post says, the post's number used for the check)
CHECKS = [
    ("llava-hf/llava-1.5-7b-hf", "LLaVA-1.5", "576 tokens", 576),
    ("llava-hf/llava-v1.6-mistral-7b-hf", "LLaVA-NeXT", "up to 2 880 tokens", 2880),
    ("HuggingFaceM4/idefics2-8b", "IDEFICS2", "64 / image (resampler)", 64),
    ("HuggingFaceM4/Idefics3-8B-Llama3", "IDEFICS3", "169 per tile (pixel shuffle)", 169),
    ("OpenGVLab/InternVL2_5-1B", "InternVL 2.5", "256 per tile (pixel shuffle)", 256),
    ("Qwen/Qwen2-VL-2B-Instruct", "Qwen2-VL", "variable, 4 to 16 384", 16384),
    ("mistral-community/pixtral-12b", "Pixtral 12B", "variable, up to 4 096", 4096),
    ("THUDM/cogvlm-chat-hf", "CogVLM", "1 225 tokens", 1225),
    ("unsloth/gemma-3-4b-it", "Gemma 3", "256 per window (pooling)", 256),
]
# Gemma 3 and PaliGemma are gated under google/. unsloth/gemma-3-4b-it is a public copy of the
# Gemma 3 4B config. PaliGemma has no ungated copy we trust, so it is not checked.


def _load(repo: str, filename: str = "config.json") -> dict:
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, filename)) as f:
        return json.load(f)


def _text_shape(repo: str, cfg: dict) -> dict | None:
    """(layers, kv_heads, head_dim) of the language model, via transformers where it knows the type."""
    try:
        if "auto_map" in cfg:          # custom-code repo: never run its code, read the JSON below
            raise LookupError
        from transformers import AutoConfig

        tc = AutoConfig.from_pretrained(repo).get_text_config()
        heads = tc.num_attention_heads
        kv = getattr(tc, "num_key_value_heads", None) or heads
        hd = getattr(tc, "head_dim", None) or tc.hidden_size // heads
        return {"layers": tc.num_hidden_layers, "kv_heads": kv, "head_dim": hd}
    except Exception:
        pass
    llm = cfg.get("llm_config") or cfg          # InternVL keeps the LM config inline; CogVLM is flat
    if "num_hidden_layers" in llm and "num_attention_heads" in llm:
        heads = llm["num_attention_heads"]
        kv = llm.get("num_key_value_heads") or heads
        return {"layers": llm["num_hidden_layers"], "kv_heads": kv, "head_dim": llm["hidden_size"] // heads}
    return None


def visual_tokens(repo: str, cfg: dict) -> tuple[int, str]:
    """Tokens per image (or per tile / window) computed from config fields, and how."""
    v = cfg.get("vision_config", {})
    mt = cfg.get("model_type")
    if mt == "llava" and v.get("model_type") == "pixtral":
        side = v["image_size"] // v["patch_size"]
        return side * side, f"({v['image_size']}/{v['patch_size']})^2 patches at the largest size, plus one [IMG_BREAK] per row"
    if mt == "llava":
        side = v["image_size"] // v["patch_size"]
        drop_cls = cfg.get("vision_feature_select_strategy") == "default"
        return side * side, f"({v['image_size']}/{v['patch_size']})^2, CLS dropped={drop_cls}"
    if mt == "llava_next":
        pins = [tuple(p) for p in cfg["image_grid_pinpoints"]]
        best = max(llava_next_tokens(p, pins, v["image_size"], v["patch_size"])["tokens"] for p in pins)
        side = v["image_size"] // v["patch_size"]
        return best, (f"largest over the {len(pins)} canvases: {side}^2 global + unpadded tiles + one newline "
                      f"token per row")
    if mt == "idefics2":
        from transformers.models.idefics2.configuration_idefics2 import Idefics2PerceiverConfig

        n = cfg.get("perceiver_config", {}).get("resampler_n_latents", Idefics2PerceiverConfig().resampler_n_latents)
        return n, "perceiver resampler latents"
    if mt == "idefics3":
        side = v["image_size"] // v["patch_size"]
        s = cfg["scale_factor"]
        return side * side // (s * s), f"({v['image_size']}/{v['patch_size']})^2 / {s}^2"
    if mt == "internvl_chat":
        side = cfg["force_image_size"] // v["patch_size"]
        return int(side * side * cfg["downsample_ratio"] ** 2), (
            f"({cfg['force_image_size']}/{v['patch_size']})^2 * {cfg['downsample_ratio']}^2, "
            f"up to {cfg['max_dynamic_patch']} tiles + thumbnail")
    if mt == "qwen2_vl":
        pre = _load(repo, "preprocessor_config.json")
        f = pre["patch_size"] * pre["merge_size"]
        lo, hi = pre["min_pixels"] // (f * f), pre["max_pixels"] // (f * f)
        return hi, f"min_pixels/{f}^2 = {lo} to max_pixels/{f}^2 = {hi}"
    if mt == "gemma3":
        side = v["image_size"] // v["patch_size"]
        return cfg["mm_tokens_per_image"], f"({v['image_size']}/{v['patch_size']})^2 = {side * side} patches pooled to mm_tokens_per_image"
    if "image_size" in v and "patch_size" in v:     # CogVLM
        side = v["image_size"] // v["patch_size"]
        return side * side, f"({v['image_size']}/{v['patch_size']})^2"
    raise ValueError(f"no rule for {repo}")


def check_all() -> list[dict]:
    rows = []
    for repo, name, claim, claimed in CHECKS:
        row = {"repo": repo, "model": name, "post_says": claim}
        try:
            cfg = _load(repo)
            tok, how = visual_tokens(repo, cfg)
            row.update(tokens_from_config=tok, how=how, matches_post=(tok == claimed))
            shape = _text_shape(repo, cfg)
            if shape:
                per = kv_bytes_per_token(**shape)
                row.update(lm_shape=shape, kv_kib_per_token_bf16=per / 1024,
                           kv_mib_for_these_tokens=round(per * tok / 2**20, 1))
        except Exception as e:  # gated repo or network failure: record it, do not guess
            row.update(error=f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")
        rows.append(row)
    return rows
