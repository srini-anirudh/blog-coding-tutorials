"""A synthetic "glyph board" task with one coarse question and one reading question.

Each 32 x 32 image has a background colour (8 choices) and a 4 x 4 board of glyphs, one per 8 x 8
cell. A glyph is one of 8 random 4 x 4 bit patterns drawn with 2 x 2 pixel bits, so it fills its cell.

    coarse question:   [Q_COLOR, FILL]  -> colour id        one fact about the whole picture
    reading question:  [Q_GLYPH, POS_i] -> glyph id at i    needs the detail of one small cell

The text is 3 tokens: the two question tokens and the answer. Only the answer is supervised. The
reading question stands in for OCR: the answer lives in a few pixels, so squeezing the image into
fewer visual tokens should hurt it far more than the coarse question.
"""

from __future__ import annotations

import torch

N_COLORS, N_GLYPHS, BOARD, CELL, BITS = 8, 8, 4, 8, 4
IMAGE = BOARD * CELL  # 32

PAD, Q_COLOR, Q_GLYPH, FILL = 0, 1, 2, 3
POS0 = 4                       # POS_i = POS0 + i, i < 16
COLOR0 = POS0 + BOARD * BOARD  # answer tokens for colours
GLYPH0 = COLOR0 + N_COLORS     # answer tokens for glyphs
VOCAB = GLYPH0 + N_GLYPHS      # 36
TEXT_LEN = 3

PALETTE = torch.tensor([
    [0.85, 0.25, 0.25], [0.25, 0.70, 0.30], [0.25, 0.35, 0.85], [0.90, 0.80, 0.20],
    [0.60, 0.30, 0.75], [0.95, 0.55, 0.15], [0.30, 0.75, 0.80], [0.75, 0.75, 0.75],
])


def glyph_bank(seed: int = 1234) -> torch.Tensor:
    """N_GLYPHS distinct 4 x 4 bit patterns, 5 to 11 bits on, at least 5 bits apart from each other."""
    gen = torch.Generator().manual_seed(seed)
    bank: list[torch.Tensor] = []
    while len(bank) < N_GLYPHS:
        g = (torch.rand(BITS, BITS, generator=gen) < 0.5).float()
        if 5 <= g.sum() <= 11 and all((g != h).sum() >= 5 for h in bank):
            bank.append(g)
    return torch.stack(bank)


BANK = glyph_bank()
# Each glyph as an 8 x 8 cell: every bit becomes a 2 x 2 pixel block. N_GLYPHS x CELL x CELL.
TILES = BANK.repeat_interleave(CELL // BITS, 1).repeat_interleave(CELL // BITS, 2)


def render_ink(glyphs: torch.Tensor) -> torch.Tensor:
    """B x 16 glyph ids -> B x 32 x 32 ink mask."""
    b = glyphs.shape[0]
    cells = TILES[glyphs]  # B x 16 x 8 x 8
    cells = cells.view(b, BOARD, BOARD, CELL, CELL).permute(0, 1, 3, 2, 4)
    return cells.reshape(b, IMAGE, IMAGE)


def make_batch(batch: int, gen: torch.Generator, task: str = "mixed") -> dict[str, torch.Tensor]:
    """Images, question tokens and targets. `task` is 'color', 'glyph' or 'mixed' (half each)."""
    colors = torch.randint(N_COLORS, (batch,), generator=gen)
    glyphs = torch.randint(N_GLYPHS, (batch, BOARD * BOARD), generator=gen)
    img = PALETTE[colors][:, :, None, None].expand(batch, 3, IMAGE, IMAGE).clone()
    img += 0.05 * torch.randn(img.shape, generator=gen)
    img *= 1 - 0.9 * render_ink(glyphs)[:, None]
    if task == "color":
        is_glyph = torch.zeros(batch, dtype=torch.bool)
    elif task == "glyph":
        is_glyph = torch.ones(batch, dtype=torch.bool)
    else:
        is_glyph = torch.rand(batch, generator=gen) < 0.5
    pos = torch.randint(BOARD * BOARD, (batch,), generator=gen)
    answer = torch.where(is_glyph, GLYPH0 + glyphs[torch.arange(batch), pos], COLOR0 + colors)
    text = torch.stack([torch.where(is_glyph, Q_GLYPH, Q_COLOR), torch.where(is_glyph, POS0 + pos, FILL), answer], 1)
    targets = torch.full_like(text, -100)
    targets[:, 1] = answer  # the logit at position 1 predicts token 2, the answer
    return {"pixels": (img - 0.5) / 0.25, "text": text, "targets": targets, "is_glyph": is_glyph}
