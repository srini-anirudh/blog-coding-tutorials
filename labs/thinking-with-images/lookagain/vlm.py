"""A thin, explicit wrapper around Qwen2-VL-2B-Instruct.

We do the image preprocessing ourselves instead of using the Hugging Face processor, for two reasons. The processor
needs torchvision, and doing it by hand shows exactly what the post calls the resolution policy: we choose the encoder
size, cut 14 px patches, and every 2 x 2 group of patches becomes one language-model token.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
PATCH, MERGE, TEMPORAL = 14, 2, 2
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"


def patchify(img: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn an RGB image whose sides are multiples of 28 into Qwen2-VL patch rows and its (t, h, w) grid."""
    w, h = img.size
    assert w % (PATCH * MERGE) == 0 and h % (PATCH * MERGE) == 0, (w, h)
    x = (np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0 - MEAN) / STD  # H, W, C
    x = np.repeat(x.transpose(2, 0, 1)[None], TEMPORAL, axis=0)  # a still image is a 2-frame clip: T, C, H, W
    gh, gw = h // PATCH, w // PATCH
    x = x.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)  # patches of one 2x2 merge group sit next to each other
    rows = x.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH)
    return torch.from_numpy(np.ascontiguousarray(rows)), torch.tensor([1, gh, gw])


def n_tokens(img: Image.Image) -> int:
    """Language-model tokens for one image: (H/14)(W/14)/4."""
    w, h = img.size
    return (w // PATCH) * (h // PATCH) // (MERGE * MERGE)


class QwenVLM:
    def __init__(self, device: str | None = None, attn: str = "sdpa", dtype=None):
        from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)  # bf16 matmuls are slow on CPU
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=dtype, attn_implementation=attn).to(self.device).eval()
        self.yes_id = self.tok.convert_tokens_to_ids("Yes")
        self.no_id = self.tok.convert_tokens_to_ids("No")
        self.image_pad = self.tok.convert_tokens_to_ids("<|image_pad|>")

    # ---- building inputs -------------------------------------------------------------------------------------
    def _text(self, images: list[Image.Image], prompt: str, answer_prefix: str = "") -> str:
        vis = "".join("<|vision_start|>" + "<|image_pad|>" * n_tokens(im) + "<|vision_end|>" for im in images)
        return f"{SYSTEM}<|im_start|>user\n{vis}{prompt}<|im_end|>\n<|im_start|>assistant\n{answer_prefix}"

    def build(self, items: list[tuple[list[Image.Image], str]], answer_prefix: str = "") -> dict:
        """Left-padded batch. Each item is (images, prompt); images may be empty (the blind probe)."""
        ids = [self.tok(self._text(ims, p, answer_prefix), add_special_tokens=False).input_ids for ims, p in items]
        return self.build_from_ids(ids, [ims for ims, _ in items])

    def build_from_ids(self, ids: list[list[int]], images: list[list[Image.Image]]) -> dict:
        L = max(len(x) for x in ids)
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([[pad] * (L - len(x)) + x for x in ids])
        mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids])
        # 1 marks image tokens. The model uses it to give them 2D (row, column) rotary positions: M-RoPE.
        out = {"input_ids": input_ids, "attention_mask": mask,
               "mm_token_type_ids": (input_ids == self.image_pad).long()}
        flat = [im for ims in images for im in ims]
        if flat:
            rows, grids = zip(*(patchify(im) for im in flat))
            out["pixel_values"] = torch.cat(rows).to(self.model.dtype)
            out["image_grid_thw"] = torch.stack(grids)
        return {k: v.to(self.device) for k, v in out.items()}

    # ---- calls -----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, items, max_new_tokens=24, sample=False, temperature=1.0, n=1, answer_prefix="", seed=None,
                 **gen_kw):
        """Greedy or sampled answers. Returns a list (per item) of lists (per sample) of strings."""
        if seed is not None:
            torch.manual_seed(seed)
        batch = self.build(items, answer_prefix)
        kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
        out = self.model.generate(**batch, max_new_tokens=max_new_tokens, num_return_sequences=n,
                                  pad_token_id=self.tok.pad_token_id, **kw, **gen_kw)
        texts = self.tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        return [texts[i * n:(i + 1) * n] for i in range(len(items))]

    def ask(self, images, prompt, **kw) -> str:
        return self.generate([(images, prompt)], **kw)[0][0]

    @torch.no_grad()
    def yes_prob(self, items) -> list[float]:
        """P(Yes) / (P(Yes) + P(No)) for the first answer token. Used as a confidence score."""
        batch = self.build(items)
        logits = self.model(**batch).logits[:, -1].float()
        two = torch.stack([logits[:, self.yes_id], logits[:, self.no_id]], -1).softmax(-1)
        return two[:, 0].tolist()

    @torch.no_grad()
    def attention_to_images(self, input_ids: list[int], images: list[Image.Image], layers=None) -> torch.Tensor:
        """For every position, the share of its attention that lands on image tokens.

        Averaged over heads and the chosen layers. Needs the model loaded with attn="eager".
        Returns a (seq_len,) tensor and leaves the image-token positions to the caller.
        """
        batch = self.build_from_ids([input_ids], [images])
        att = self.model(**batch, output_attentions=True).attentions
        layers = layers if layers is not None else range(len(att))
        is_img = (batch["input_ids"][0] == self.image_pad).float()
        share = torch.stack([(att[i][0].float() @ is_img).mean(0) for i in layers]).mean(0)
        return share.cpu()

    @torch.no_grad()
    def attention_map(self, image: Image.Image, prompt: str, layers=None) -> torch.Tensor:
        """Attention from the last prompt position to each visual token, as a (h/2, w/2) map."""
        batch = self.build([([image], prompt)])
        att = self.model(**batch, output_attentions=True).attentions
        layers = layers if layers is not None else range(len(att) // 2, len(att))
        pos = (batch["input_ids"][0] == self.image_pad).nonzero().squeeze(1)
        a = torch.stack([att[i][0, :, -1, pos].float().mean(0) for i in layers]).mean(0)
        _, gh, gw = batch["image_grid_thw"][0].tolist()
        return a.reshape(gh // MERGE, gw // MERGE).cpu()
