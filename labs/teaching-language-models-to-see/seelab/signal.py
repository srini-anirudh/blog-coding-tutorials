"""Where each training objective puts its learning signal (the post's Figure 9), measured as gradients.

All three objectives are differentiated with respect to the same tensor: SigLIP-base's 14 x 14 grid of patch
features (the encoder's last hidden state, before any pooling). The gradient norm at each patch is the
signal that patch would receive if the encoder were trained.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from . import common  # noqa: F401  (puts C7 on sys.path)
from tinyvlm.data import COLORS, SHAPES, draw_shape  # noqa: E402
from tinyvlm.merge import merge_masked_scatter  # noqa: E402
from tinyvlm.positions import position_ids_1d  # noqa: E402
from tinyvlm.projector import pixel_shuffle  # noqa: E402


def two_object_scene(seed: int, size: int = 224) -> dict:
    """One shape on the left half, a different one on the right half; masks for each on the 14 x 14 grid."""
    rng = random.Random(seed)
    colors = rng.sample(list(COLORS), 2)
    shapes = [rng.choice(SHAPES), rng.choice(SHAPES)]
    shade = rng.randint(235, 255)
    img = Image.new("RGB", (size, size), (shade, shade, shade))
    masks = []
    for k, (color, shape) in enumerate(zip(colors, shapes)):
        r = rng.randint(20, 34)
        cx = rng.randint(r + 6, size // 2 - r - 6) + k * size // 2
        cy = rng.randint(r + 6, size - r - 6)
        draw_shape(ImageDraw.Draw(img), shape, cx, cy, r, COLORS[color])
        m = Image.new("L", (size, size), 0)
        draw_shape(ImageDraw.Draw(m), shape, cx, cy, r, 255)
        a = np.asarray(m) > 0
        masks.append(torch.from_numpy(a.reshape(14, 16, 14, 16).mean((1, 3)) >= 0.1).flatten())
    return {"image": img, "objects": list(zip(colors, shapes)), "masks": masks}


def patch_features(vision, pixel_values: torch.Tensor) -> torch.Tensor:
    """n x 196 x 768, a leaf we can take gradients with respect to."""
    with torch.no_grad():
        h = vision(pixel_values=pixel_values).last_hidden_state
    return h.detach().requires_grad_(True)


def siglip_text(siglip, tokenizer_file: str, texts: list[str]) -> torch.Tensor:
    """Normalised SigLIP text vectors. SigLIP's tokenizer: lower case, no punctuation, EOS, pad with EOS to 64."""
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    clean = ["".join(c for c in t.lower() if c.isalnum() or c == " ").strip() for t in texts]
    ids = [tok(t, add_special_tokens=False).input_ids[:63] + [1] for t in clean]
    ids = torch.tensor([i + [1] * (64 - len(i)) for i in ids])
    with torch.no_grad():
        out = siglip.get_text_features(input_ids=ids)
    return F.normalize(getattr(out, "pooler_output", out), dim=-1)


def siglip_logits(siglip, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """n images x m texts: t cos + b, with the image pooled by SigLIP's attention head."""
    u = F.normalize(siglip.vision_model.head(h), dim=-1)
    return siglip.logit_scale.exp() * u @ v.T + siglip.logit_bias


def grad_contrastive(siglip, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """SigLIP's own objective for each matching pair (row i, text i): -log sigmoid(t cos + b)."""
    logit = siglip_logits(siglip, h, v).diagonal()
    (g,) = torch.autograd.grad(-F.logsigmoid(logit).sum(), h)
    return g


def grad_next_token(vlm, h: torch.Tensor, batch: dict) -> torch.Tensor:
    """The VLM's caption loss, from the patch features through pixel shuffle, connector and frozen LM."""
    feats = vlm.projector(pixel_shuffle(h, vlm.config.pixel_shuffle_factor))
    ids = batch["input_ids"]
    embeds = merge_masked_scatter(vlm.get_input_embeddings()(ids), ids, feats, vlm.config.image_token_id)
    out = vlm.lm(inputs_embeds=embeds, attention_mask=batch["attention_mask"],
                 position_ids=position_ids_1d(batch["attention_mask"]), labels=batch["labels"])
    (g,) = torch.autograd.grad(out.loss * ids.shape[0], h)
    return g


def grad_patch_loss(h: torch.Tensor, pixel_values: torch.Tensor, decoder: torch.nn.Linear) -> torch.Tensor:
    """An auxiliary per-patch target (AIMv2- or MAE-style): regress each patch's own 16 x 16 x 3 pixels."""
    n = pixel_values.shape[0]
    target = pixel_values.reshape(n, 3, 14, 16, 14, 16).permute(0, 2, 4, 1, 3, 5).reshape(n, 196, 768)
    loss = F.mse_loss(decoder(h), target, reduction="sum") / 196
    (g,) = torch.autograd.grad(loss, h)
    return g


def shares(g: torch.Tensor, masks: list[torch.Tensor]) -> dict[str, float]:
    """Fraction of total per-patch gradient norm that lands on each mask, and on everything else."""
    norm = g.norm(dim=-1)  # 196
    total = norm.sum()
    out, rest = {}, torch.ones_like(masks[0])
    for k, m in enumerate(masks):
        out[f"obj{k}"] = (norm[m].sum() / total).item()
        rest = rest & ~m
    out["background"] = (norm[rest].sum() / total).item()
    out["max_over_mean"] = (norm.max() / norm.mean()).item()
    return out
