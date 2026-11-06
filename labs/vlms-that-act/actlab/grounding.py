"""Grounding small icons on a large screen, and why zooming in helps.

A screen is S x S pixels with toolbars of 12-pixel icons. The instruction names one glyph ("click the gear"), and
that glyph appears exactly once. A small fully convolutional grounder turns the resized screenshot into a grid of
visual tokens (one token per 8 x 8 input pixels) and points at one token, GUI-Actor style, plus a sub-token offset.
A click is correct if it lands inside the target icon's box, as in ScreenSpot.

The same weights are run three ways: one pass at a small budget, one pass at a larger budget, and a zoom search that
tiles the screen and runs the small budget on each tile. A "no target here" logit lets the search pick a tile.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCREEN = 384
ICON = 12
PATCH = 8  # input pixels per visual token, along each side

_G = [
    "..##..|.####.|######|######|.####.|..##..",  # disc
    "##..##|##..##|..##..|..##..|##..##|##..##",  # x
    "######|#....#|#....#|#....#|#....#|######",  # square outline
    "..##..|..##..|######|######|..##..|..##..",  # plus
    "#.....|##....|###...|####..|#####.|######",  # triangle
    "######|######|......|......|######|######",  # two bars
    "#.#.#.|.#.#.#|#.#.#.|.#.#.#|#.#.#.|.#.#.#",  # checker
    "..#...|..##..|######|######|..##..|..#...",  # arrow
]
GLYPHS = [np.kron(np.array([[c == "#" for c in row] for row in g.split("|")], bool), np.ones((2, 2), bool))
          for g in _G]
N_GLYPHS = len(GLYPHS)
_INK = [(30, 30, 30), (40, 60, 150), (150, 40, 40), (30, 110, 60), (90, 50, 120)]


def make_screen(rng: random.Random, size: int = SCREEN):
    """A screen with 3 to 6 toolbars of tightly packed icons. Returns (image HxWx3 uint8, target box, glyph id)."""
    img = np.full((size, size, 3), 238, np.uint8)
    for _ in range(3):  # a few flat panels, like real app chrome
        x, y = rng.randrange(0, size - 60), rng.randrange(0, size - 60)
        img[y:y + rng.randint(30, 120), x:x + rng.randint(30, 160)] = rng.choice([(225, 228, 235), (250, 250, 250)])
    target = rng.randrange(N_GLYPHS)
    slots = []
    rows = rng.sample(range(4, size - ICON - 4, 18), rng.randint(3, 6))
    for y in rows:
        x0 = rng.randrange(0, size // 2)
        n = rng.randint(8, (size - x0) // (ICON + 2))
        slots += [(x0 + i * (ICON + 2), y) for i in range(n)]
    t_slot = rng.randrange(len(slots))
    for i, (x, y) in enumerate(slots):
        g = target if i == t_slot else rng.choice([k for k in range(N_GLYPHS) if k != target])
        img[y:y + ICON, x:x + ICON][GLYPHS[g]] = rng.choice(_INK)
    tx, ty = slots[t_slot]
    return img, (tx, ty, ICON, ICON), target


def to_input(img: np.ndarray, crop: tuple[int, int, int], r: int) -> torch.Tensor:
    """Crop (x, y, side) out of the screen and area-resize it to r x r: the model's visual-token budget."""
    x, y, c = crop
    t = torch.from_numpy(img[y:y + c, x:x + c]).permute(2, 0, 1).float().div(255.0)
    return F.interpolate(t[None], size=(r, r), mode="area")[0]


def label_for(box, crop, r):
    """Token index (or the null index) and sub-token offset for a target box inside a crop."""
    x, y, c = crop
    g = r // PATCH
    bx, by, bw, bh = box
    if not (bx >= x and by >= y and bx + bw <= x + c and by + bh <= y + c):
        return g * g, (0.0, 0.0)
    cell = c / g
    u, v = (bx + bw / 2 - x) / cell, (by + bh / 2 - y) / cell
    col, row = min(int(u), g - 1), min(int(v), g - 1)
    return row * g + col, (u - col - 0.5, v - row - 0.5)


class Grounder(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(5, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, ch, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(ch, ch, 3, 2, 1), nn.GELU())  # stride 8: one cell per visual token
        self.query = nn.Embedding(N_GLYPHS, 2 * ch)
        self.mix = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(), nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU())
        self.point = nn.Conv2d(ch, 3, 1)  # pointer logit and (dx, dy) per token
        self.null = nn.Linear(ch, 1)

    def forward(self, x, q):
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        x = torch.cat([x, xx.expand(b, 1, h, w), yy.expand(b, 1, h, w)], 1)
        f = self.stem(x)
        gamma, beta = self.query(q)[:, :, None, None].chunk(2, 1)
        f = self.mix(f * (1 + gamma) + beta)
        out = self.point(f)
        logits = out[:, 0].flatten(1)
        null = self.null(f.amax((2, 3)))
        return torch.cat([logits, null], 1), torch.tanh(out[:, 1:].flatten(2).transpose(1, 2)) * 0.5


# (crop side, input resolution, chance the crop contains the target) for training
TRAIN_MIX = [(384, 48, 1.0), (384, 96, 1.0), (384, 192, 1.0), (192, 48, 0.6), (96, 48, 0.6)]


def sample_batch(rng: random.Random, n: int, c: int, r: int, p_inside: float):
    xs, qs, ys, offs = [], [], [], []
    for _ in range(n):
        img, box, g = make_screen(rng)
        if c == SCREEN:
            crop = (0, 0, c)
        elif rng.random() < p_inside:
            bx, by = box[0], box[1]
            crop = (rng.randint(max(0, bx + ICON - c), min(bx, SCREEN - c)),
                    rng.randint(max(0, by + ICON - c), min(by, SCREEN - c)), c)
        else:
            crop = (rng.randint(0, SCREEN - c), rng.randint(0, SCREEN - c), c)
        idx, off = label_for(box, crop, r)
        xs.append(to_input(img, crop, r)), qs.append(g), ys.append(idx), offs.append(off)
    return torch.stack(xs), torch.tensor(qs), torch.tensor(ys), torch.tensor(offs)


def train_grounder(steps: int = 3000, batch: int = 48, seed: int = 0, log_every: int = 500, log=print) -> Grounder:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = Grounder()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    weights = [1, 1, 2, 2, 2]
    for step in range(1, steps + 1):
        c, r, p_in = rng.choices(TRAIN_MIX, weights)[0]
        x, q, y, off = sample_batch(rng, batch, c, r, p_in)
        logits, offsets = model(x, q)
        loss = F.cross_entropy(logits, y)
        has = y < logits.shape[1] - 1
        if has.any():
            pred = offsets[has, y[has]]
            loss = loss + 2.0 * F.l1_loss(pred, off[has])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log_every and step % log_every == 0:
            log(f"step {step:5d}  crop {c:3d} -> {r:3d}px  loss {loss.item():.3f}")
    return model.eval()


@torch.no_grad()
def predict(model: Grounder, img, glyph: int, crop, r: int):
    """Return (click x, click y in screen pixels, confidence that the target is in this crop)."""
    x0, y0, c = crop
    g = r // PATCH
    logits, offsets = model(to_input(img, crop, r)[None], torch.tensor([glyph]))
    p = logits.softmax(1)[0]
    k = int(p[:-1].argmax())
    dx, dy = offsets[0, k].tolist()
    cell = c / g
    return x0 + (k % g + 0.5 + dx) * cell, y0 + (k // g + 0.5 + dy) * cell, float(1 - p[-1])


def tiles(side: int, size: int = SCREEN, overlap: int = ICON):
    """Overlapping square tiles so that every icon lies whole inside at least one tile."""
    n = 1 if side >= size else math.ceil((size - side) / (side - overlap)) + 1
    starts = [round(i * (size - side) / max(n - 1, 1)) for i in range(n)]
    return [(x, y, side) for y in starts for x in starts]


def zoom_search(model, img, glyph, tile_side: int, r: int = 48):
    best = max((predict(model, img, glyph, t, r) for t in tiles(tile_side)), key=lambda z: z[2])
    return best[0], best[1]


def hit(box, x, y) -> bool:
    bx, by, bw, bh = box
    return bx <= x < bx + bw and by <= y < by + bh


def evaluate(model, n: int = 500, seed: int = 123) -> dict:
    """Click accuracy for single passes at three budgets and for zoom search at the smallest budget."""
    rng = random.Random(seed)
    screens = [make_screen(rng) for _ in range(n)]
    out = {}
    for r in (48, 96, 192):
        acc = np.mean([hit(b, *predict(model, im, g, (0, 0, SCREEN), r)[:2]) for im, b, g in screens])
        g = r // PATCH
        out[f"single pass, {r}px input"] = dict(tokens=g * g, px_per_token=SCREEN / g, accuracy=float(acc))
    for side in (192, 96):
        acc = np.mean([hit(b, *zoom_search(model, im, g, side)) for im, b, g in screens])
        nt = len(tiles(side))
        out[f"zoom search, {nt} tiles of {side}px at 48px"] = dict(
            tokens=nt * 36, px_per_token=side / 6, accuracy=float(acc))
    return out
