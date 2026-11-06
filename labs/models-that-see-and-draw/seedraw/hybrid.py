"""A Transfusion-style hybrid: one transformer, a next-token loss on text and a flow loss on image patches.

Two sequence layouts, both 21 positions long:

    draw:  <bos> shape side <boi> p1 ... p16 <eoi>      text first, the image is generated
    read:  <bos> <boi> p1 ... p16 <eoi> shape side      image first, the caption is generated

The attention mask is causal across the sequence, but the 16 patches of one image see each other in both
directions (`transfusion_mask`). Image patches are noised on the straight line of rectified flow and the trunk
predicts their velocity. The total loss is L = L_LM + lambda * L_flow, with lambda = 5 as in Transfusion.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .data import BOI, BOS, EOI, WORDS, caption_tokens, from_patches, to_patches
from .flow import flow_target, noisy_point, time_features
from .transformer import Trunk

N_PATCH, PATCH_DIM = 16, 16
SEQ = 21
DRAW_IMG = slice(4, 20)  # patch positions in the draw layout
READ_IMG = slice(2, 18)  # patch positions in the read layout


def transfusion_mask(n: int, image_spans: list[tuple[int, int]], bidirectional_images: bool = True) -> torch.Tensor:
    """allowed[i, j]: causal everywhere, plus full attention inside each [start, end) image span."""
    allowed = torch.ones(n, n, dtype=torch.bool).tril()
    if bidirectional_images:
        for s, e in image_spans:
            allowed[s:e, s:e] = True
    return allowed


class HybridModel(nn.Module):
    def __init__(self, d: int = 96, layers: int = 3, bidirectional_images: bool = True):
        super().__init__()
        self.text_in = nn.Embedding(len(WORDS), d)
        self.patch_in = nn.Linear(PATCH_DIM, d)
        self.time_in = nn.Linear(16, d)
        self.trunk = Trunk(SEQ, d, layers)
        self.text_out = nn.Linear(d, len(WORDS))
        self.patch_out = nn.Linear(d, PATCH_DIM)
        self.draw_mask = transfusion_mask(SEQ, [(DRAW_IMG.start, DRAW_IMG.stop)], bidirectional_images)
        self.read_mask = transfusion_mask(SEQ, [(READ_IMG.start, READ_IMG.stop)], bidirectional_images)

    def _embed(self, text: torch.Tensor, patches: torch.Tensor, t: torch.Tensor, img: slice) -> torch.Tensor:
        """text: (n, 5) ids in sequence order around the image; patches: (n, 16, 16); t: (n,)."""
        n = text.shape[0]
        h_text = self.text_in(text)
        h_img = self.patch_in(patches) + self.time_in(time_features(t))[:, None]
        return torch.cat([h_text[:, : img.start], h_img, h_text[:, img.start :]], dim=1)

    def forward_draw(self, caption: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
        n = caption.shape[0]
        text = torch.cat([torch.full((n, 1), BOS), caption, torch.full((n, 1), BOI), torch.full((n, 1), EOI)], 1)
        out = self.trunk(self._embed(text, xt, t, DRAW_IMG), self.draw_mask)
        return self.text_out(out[:, :3]), self.patch_out(out[:, DRAW_IMG])  # text logits predict caption + <boi>

    def forward_read(self, caption: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
        n = caption.shape[0]
        text = torch.cat([torch.full((n, 1), BOS), torch.full((n, 1), BOI), torch.full((n, 1), EOI), caption], 1)
        out = self.trunk(self._embed(text, xt, t, READ_IMG), self.read_mask)
        return self.text_out(out[:, 18:20]), self.patch_out(out[:, READ_IMG])  # <eoi> -> shape, shape -> side


def hybrid_losses(model: HybridModel, images: torch.Tensor, shape: torch.Tensor, side: torch.Tensor,
                  read_t_max: float = 1.0) -> dict[str, torch.Tensor]:
    """Both layouts on one batch. Returns the text loss, the flow loss and the read-only caption loss."""
    n = images.shape[0]
    x0 = to_patches(images)
    cap = caption_tokens(shape, side)
    out = {}

    t = torch.rand(n)
    eps = torch.randn_like(x0)
    text_logits, v = model.forward_draw(cap, noisy_point(x0, t, eps), t)
    draw_targets = torch.cat([cap, torch.full((n, 1), BOI)], 1)
    lm_draw = F.cross_entropy(text_logits.flatten(0, 1), draw_targets.flatten())
    flow_draw = ((v - flow_target(x0, eps)) ** 2).mean()

    t_read = torch.rand(n) * read_t_max  # Transfusion caps this noise when the image comes before its caption
    eps_r = torch.randn_like(x0)
    text_logits_r, v_r = model.forward_read(cap, noisy_point(x0, t_read, eps_r), t_read)
    lm_read = F.cross_entropy(text_logits_r.flatten(0, 1), cap.flatten())
    flow_read = ((v_r - flow_target(x0, eps_r)) ** 2).mean()

    out["lm"] = (lm_draw + lm_read) / 2
    out["flow"] = (flow_draw + flow_read) / 2
    out["lm_read"] = lm_read
    return out


def train_hybrid(model: HybridModel, data: dict, steps: int, lam: float = 5.0, batch: int = 128, lr: float = 1e-3,
                 read_t_max: float = 1.0, seed: int = 0, eval_fn=None, eval_every: int = 0) -> list[dict]:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    log = []
    for step in range(1, steps + 1):
        i = torch.randint(0, len(data["images"]), (batch,))
        losses = hybrid_losses(model, data["images"][i], data["shape"][i], data["side"][i], read_t_max)
        loss = losses["lm"] + lam * losses["flow"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if eval_fn is not None and eval_every and step % eval_every == 0:
            log.append({"step": step, "lm": losses["lm"].item(), "flow": losses["flow"].item(), **eval_fn(model)})
    return log


@torch.no_grad()
def draw(model: HybridModel, shape: torch.Tensor, side: torch.Tensor, steps: int = 30) -> torch.Tensor:
    """Text-to-image: append pure-noise patches after the caption and integrate the flow."""
    n = shape.shape[0]
    cap = caption_tokens(shape, side)
    x = torch.randn(n, N_PATCH, PATCH_DIM)
    dt = 1.0 / steps
    for k in range(steps, 0, -1):
        t = torch.full((n,), k * dt)
        _, v = model.forward_draw(cap, x, t)
        x = x - dt * v
    return from_patches(x).clamp(0, 1)


@torch.no_grad()
def read(model: HybridModel, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Image-to-text on a clean image: greedy shape word, then greedy side word."""
    n = images.shape[0]
    x0 = to_patches(images)
    t = torch.zeros(n)
    dummy = torch.full((n, 2), 4)
    logits, _ = model.forward_read(dummy, x0, t)
    shape = logits[:, 0, 4:8].argmax(-1)
    cap = torch.stack([4 + shape, torch.full_like(shape, 8)], 1)
    logits, _ = model.forward_read(cap, x0, t)
    side = logits[:, 1, 8:10].argmax(-1)
    return shape, side
