"""Qwen2-VL taken apart into the five stages of the post: preprocess, encode, (reduce), prefill, decode.

Qwen2-VL uses 14-pixel patches and a 2 x 2 merge, the rule behind the post's worked example, so its real token
counts can be checked against `costmodel`. We call the encoder and the language model ourselves so that visual
tokens can be dropped or merged between them while each kept token keeps its original 3D (M-RoPE) position.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .selection import Reduced, keep_all

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def load(model_id: str = MODEL_ID, device: str = "cpu", attn: str = "sdpa"):
    from transformers import Qwen2VLForConditionalGeneration
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype, attn_implementation=attn)
    return model.to(device).eval(), load_processor(model_id)


def load_processor(model_id: str = MODEL_ID):
    """Tokenizer plus the PIL image processor (the stock processor also wants torchvision for video)."""
    import json

    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor, AutoTokenizer
    template = json.load(open(hf_hub_download(model_id, "chat_template.json")))["chat_template"]
    return Processor(AutoImageProcessor.from_pretrained(model_id, backend="pil"),
                     AutoTokenizer.from_pretrained(model_id), template)


class Processor:
    """What Qwen2-VL's processor does for one image: render the chat, run the image processor, and expand the
    single <|image_pad|> into one placeholder per visual token (patches / merge^2)."""

    def __init__(self, image_processor, tokenizer, chat_template: str, merge: int = 2):
        self.image_processor, self.tokenizer, self.template, self.merge = image_processor, tokenizer, chat_template, merge

    def __call__(self, image, question: str, device="cpu") -> dict:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.tokenizer.apply_chat_template(messages, chat_template=self.template, tokenize=False,
                                                  add_generation_prompt=True)
        img = self.image_processor(images=[image], return_tensors="pt")
        n = int(img["image_grid_thw"][0].prod()) // self.merge ** 2
        text = text.replace("<|image_pad|>", "<|image_pad|>" * n, 1)
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "mm_token_type_ids": (ids == pad).int(),
                 "pixel_values": img["pixel_values"], "image_grid_thw": img["image_grid_thw"]}
        return {k: v.to(device) for k, v in batch.items()}


def build_inputs(processor, image, question: str, device="cpu") -> dict:
    return processor(image, question, device)


def sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@dataclass
class Prepared:
    embeds: torch.Tensor      # 1 x L x d, image features already scattered in
    positions: torch.Tensor   # 3 x 1 x L
    image_slice: slice        # where the visual tokens sit in the sequence
    grid: tuple[int, int]     # visual-token grid (rows, cols) after the 2 x 2 merge


@torch.inference_mode()
def encode(model, inputs) -> torch.Tensor:
    """Vision encoder plus merger: V x d_LLM visual tokens for one image."""
    out = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
    return torch.cat(out.pooler_output, 0)


@torch.inference_mode()
def prepare(model, inputs, image_embeds: torch.Tensor) -> Prepared:
    ids = inputs["input_ids"]
    embeds = model.model.get_input_embeddings()(ids)
    is_img = (ids[0] == model.config.image_token_id)
    embeds[0, is_img] = image_embeds.to(embeds.dtype)
    positions, _ = model.model.get_rope_index(ids, inputs["mm_token_type_ids"], inputs["image_grid_thw"],
                                              attention_mask=inputs.get("attention_mask"))
    where = is_img.nonzero().flatten()
    t, h, w = inputs["image_grid_thw"][0].tolist()
    m = model.config.vision_config.spatial_merge_size
    return Prepared(embeds, positions, slice(int(where[0]), int(where[-1]) + 1), (h // m, w // m))


def apply_reduction(p: Prepared, red: Reduced) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the visual block by `red`: features in, positions taken from `red.index`."""
    s = p.image_slice
    vis_pos = p.positions[:, :, s][:, :, red.index.to(p.positions.device)]
    embeds = torch.cat([p.embeds[:, :s.start], red.features[None].to(p.embeds), p.embeds[:, s.stop:]], 1)
    positions = torch.cat([p.positions[:, :, :s.start], vis_pos, p.positions[:, :, s.stop:]], 2)
    return embeds, positions


def reduce(p: Prepared, policy=keep_all, k: int | None = None, **kw) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = p.grid
    feats = p.embeds[0, p.image_slice]
    return apply_reduction(p, policy(feats, h, w, k, **kw))


@torch.inference_mode()
def prefill(model, embeds, positions, cache=None):
    """Run the language model over the prompt. Returns last-position logits and the KV cache."""
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config.text_config) if cache is None else cache
    out = model.model.language_model(inputs_embeds=embeds, position_ids=positions, past_key_values=cache,
                                     use_cache=True)
    return model.lm_head(out.last_hidden_state[:, -1]), out.past_key_values


@torch.inference_mode()
def decode(model, logits, cache, next_pos: int, max_new_tokens: int, eos: set[int] | None = None) -> list[int]:
    """Greedy decoding. Text after the prompt continues 1D: all three M-RoPE axes advance together."""
    tokens = []
    embed = model.model.get_input_embeddings()
    for step in range(max_new_tokens):
        tok = int(logits.argmax(-1))
        tokens.append(tok)
        if eos and tok in eos:
            break
        pos = torch.full((3, 1, 1), next_pos + step, device=logits.device, dtype=torch.long)
        out = model.model.language_model(inputs_embeds=embed(torch.tensor([[tok]], device=logits.device)),
                                         position_ids=pos, past_key_values=cache, use_cache=True)
        logits = model.lm_head(out.last_hidden_state[:, -1])
    return tokens


def eos_ids(model) -> set[int]:
    eos = model.generation_config.eos_token_id
    return set(eos if isinstance(eos, list) else [eos])


def answer(model, processor, embeds, positions, max_new_tokens: int = 8) -> str:
    logits, cache = prefill(model, embeds, positions)
    toks = decode(model, logits, cache, int(positions.max()) + 1, max_new_tokens, eos_ids(model))
    return processor.tokenizer.decode(toks, skip_special_tokens=True)


class _Stop(Exception):
    pass


@torch.inference_mode()
def attention_received(model, embeds, positions, layer: int = 2) -> torch.Tensor:
    """FastV's score: attention from the last token to every token at `layer`, averaged over heads.

    Computed from the layer's own q and k projections (with RoPE), so it works with fused attention kernels,
    which never write the attention matrix out. The forward pass stops at that layer.
    """
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb
    attn = model.model.language_model.layers[layer].self_attn
    got = {}

    def hook(module, args, kwargs):
        x = kwargs["hidden_states"]
        cos, sin = kwargs["position_embeddings"]
        b, n, _ = x.shape
        q = module.q_proj(x[:, -1:]).view(b, 1, -1, module.head_dim).transpose(1, 2)
        k = module.k_proj(x).view(b, n, -1, module.head_dim).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, -1:], sin[:, -1:])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        k = k.repeat_interleave(module.num_key_value_groups, dim=1)
        got["a"] = ((q @ k.transpose(-1, -2)).float() * module.scaling).softmax(-1)[0, :, 0].mean(0)
        raise _Stop

    handle = attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        model.model.language_model(inputs_embeds=embeds, position_ids=positions, use_cache=False)
    except _Stop:
        pass
    finally:
        handle.remove()
    return got["a"]


@torch.inference_mode()
def time_stages(model, processor, image, question: str, decode_steps: int = 32, repeats: int = 3) -> dict:
    """Median wall-clock time of each stage for one request, plus the real KV-cache size."""
    device = next(model.parameters()).device
    rows = {"preprocess_s": [], "encode_s": [], "prefill_s": [], "decode_step_s": []}
    for _ in range(repeats + 1):                        # the first pass is a warm-up and is discarded
        t0 = time.perf_counter()
        inputs = build_inputs(processor, image, question, device)
        sync(device); t1 = time.perf_counter()
        feats = encode(model, inputs)
        sync(device); t2 = time.perf_counter()
        p = prepare(model, inputs, feats)
        logits, cache = prefill(model, p.embeds, p.positions)
        sync(device); t3 = time.perf_counter()
        decode(model, logits, cache, int(p.positions.max()) + 1, decode_steps)  # no EOS: fixed step count
        sync(device); t4 = time.perf_counter()
        for key, val in zip(rows, [t1 - t0, t2 - t1, t3 - t2, (t4 - t3) / decode_steps]):
            rows[key].append(val)
    out = {k: sorted(v[1:])[len(v[1:]) // 2] for k, v in rows.items()}
    _, cache = prefill(model, p.embeds, p.positions)
    kv = sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values))
    out.update(patches=int(inputs["image_grid_thw"][0].prod()), visual_tokens=int(feats.shape[0]),
               prompt_tokens=int(p.embeds.shape[1]), kv_bytes=kv)
    return out
