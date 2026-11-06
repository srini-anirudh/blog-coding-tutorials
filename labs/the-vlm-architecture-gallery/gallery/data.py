"""Synthetic glyph pages: a stand-in for "dense text" and "photographs".

An image is a 16 x 16 grey page cut into a 4 x 4 grid of 4 x 4 cells. Each cell holds one of
8 glyphs (fixed random 4 x 4 binary patterns) plus Gaussian noise. With patch size 4, one patch
is one cell, so the encoder sees 16 patches.

Two visual questions, chosen to pull the connectors apart:

    fine    "[CELL] r c"  -> the glyph in row r, column c.  Needs every cell kept apart, like
            reading small text. Chance is 1/8.
    coarse  "[MAJ]"       -> the glyph that fills half the page. A summary of the whole
            image, like naming the main object in a photo. Chance is 1/8.

The language model is pretrained first on text-only versions of the same questions, where the
page is given as 16 glyph tokens, plus an unrelated text task (addition mod 8) used to check
whether multimodal training damaged the language model.
"""

from __future__ import annotations

import torch

GRID, CELL, K = 4, 4, 8
IMG = GRID * CELL

# vocabulary
GLYPH0 = 0                 # tokens 0..7   glyph ids (also the answers to visual questions)
DIGIT0 = 8                 # tokens 8..15  digits 0..7 (rows and columns use 0..3)
CELL_Q, MAJ_Q, ADD_Q = 16, 17, 18
VOCAB = 19


def glyph_templates(seed: int = 0) -> torch.Tensor:
    """K distinct 4 x 4 binary patterns, at least 4 pixels apart from each other."""
    g = torch.Generator().manual_seed(seed)
    out: list[torch.Tensor] = []
    while len(out) < K:
        t = (torch.rand(CELL, CELL, generator=g) > 0.5).float()
        if all((t != o).sum() >= 4 for o in out):
            out.append(t)
    return torch.stack(out)


class GlyphPages:
    def __init__(self, noise: float = 0.35, seed: int = 0):
        self.templates = glyph_templates(seed)
        self.noise = noise

    def layout(self, b: int, task: str, g: torch.Generator) -> torch.Tensor:
        """B x 16 glyph ids, row-major."""
        ids = torch.randint(0, K, (b, GRID * GRID), generator=g)
        if task == "coarse":
            major = torch.randint(0, K, (b, 1), generator=g)
            # exactly half the cells hold the major glyph, so the answer is never a tie in practice
            fill = torch.rand(b, GRID * GRID, generator=g).argsort(1) < GRID * GRID // 2
            ids = torch.where(fill, major.expand_as(ids), ids)
        return ids

    def render(self, ids: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        """B x 16 glyph ids -> B x 1 x 16 x 16 images."""
        b = ids.shape[0]
        cells = self.templates[ids].view(b, GRID, GRID, CELL, CELL)
        img = cells.permute(0, 1, 3, 2, 4).reshape(b, 1, IMG, IMG)
        return img + self.noise * torch.randn(img.shape, generator=g)

    def question(self, ids: torch.Tensor, task: str, g: torch.Generator):
        """Question token ids (B x q) and answer ids (B,) for a layout."""
        b = ids.shape[0]
        if task == "fine":
            pos = torch.randint(0, GRID * GRID, (b,), generator=g)
            q = torch.stack([torch.full((b,), CELL_Q), DIGIT0 + pos // GRID, DIGIT0 + pos % GRID], 1)
            return q, ids[torch.arange(b), pos]
        if task == "coarse":
            return torch.full((b, 1), MAJ_Q), torch.mode(ids, dim=1).values
        raise ValueError(task)

    def visual_batch(self, b: int, task: str, g: torch.Generator):
        ids = self.layout(b, task, g)
        q, ans = self.question(ids, task, g)
        return self.render(ids, g), q, ans

    def text_batch(self, b: int, task: str, g: torch.Generator):
        """Text-only tasks. 'fine' and 'coarse' give the page as glyph tokens; 'add' is a + b mod 8."""
        if task == "add":
            a, c = torch.randint(0, K, (b,), generator=g), torch.randint(0, K, (b,), generator=g)
            q = torch.stack([torch.full((b,), ADD_Q), DIGIT0 + a, DIGIT0 + c], 1)
            return q, DIGIT0 + (a + c) % K
        ids = self.layout(b, task, g)
        q, ans = self.question(ids, task, g)
        return torch.cat([GLYPH0 + ids, q], 1), ans
