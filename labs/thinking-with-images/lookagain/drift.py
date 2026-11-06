"""Long generations drift away from the image: attention share on image tokens along a trace, and re-injection.

We let the model write a long description, then run one teacher-forced forward pass over prompt + trace and read
off, for every generated position, the share of attention (averaged over heads and layers) that lands on image
tokens. Re-injection inserts the plaque crop as a second image partway through the same trace and measures the
share again for the positions after it. The text is identical in both passes, so only the image placement differs.
"""
from __future__ import annotations

import torch

from .scene import Scene
from .views import View, cell_view, full_view, render
from .vlm import n_tokens

PROMPT = ("Describe this street photo in as much detail as you can: every building, object, sign and the text on "
          "it. Then say when the bakery closes.")


def vision_ids(tok, img) -> list[int]:
    return tok("<|vision_start|>" + "<|image_pad|>" * n_tokens(img) + "<|vision_end|>",
               add_special_tokens=False).input_ids


def drift_one(vlm, scene: Scene, look: View | None = None, gen_tokens: int = 384, reinject_at: int = 128):
    look = look or full_view(896)
    img = render(scene, look)
    trace = vlm.generate([([img], PROMPT)], max_new_tokens=gen_tokens, min_new_tokens=gen_tokens,
                         repetition_penalty=1.05)[0][0]  # min_new_tokens: keep writing past the natural end
    tok = vlm.tok
    prompt_ids = tok(vlm._text([img], PROMPT), add_special_tokens=False).input_ids
    trace_ids = tok(trace, add_special_tokens=False).input_ids[:gen_tokens]
    base = vlm.attention_to_images(prompt_ids + trace_ids, [img])[len(prompt_ids):]

    crop = render(scene, cell_view(*scene.plaque_cell))
    k = min(reinject_at, len(trace_ids))
    vis = vision_ids(tok, crop)
    ids2 = prompt_ids + trace_ids[:k] + vis + trace_ids[k:]
    share2 = vlm.attention_to_images(ids2, [img, crop])
    re = torch.cat([share2[len(prompt_ids):len(prompt_ids) + k], share2[len(prompt_ids) + k + len(vis):]])

    n_img = n_tokens(img)
    uniform = torch.tensor([n_img / (len(prompt_ids) + t + 1) for t in range(len(trace_ids))])
    return {"trace": trace, "base": base, "reinjected": re, "uniform": uniform, "k": k,
            "prompt_len": len(prompt_ids), "n_image_tokens": n_img}


def binned(x: torch.Tensor, width: int) -> list[float]:
    return [x[i:i + width].mean().item() for i in range(0, len(x) - width + 1, width)]
