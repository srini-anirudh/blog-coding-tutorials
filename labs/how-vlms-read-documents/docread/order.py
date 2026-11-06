"""Reading order: sort by top edge, recursive XY-cut, a margin rule for page furniture, and a small learned orderer.

Section "Reading order and long documents" and Figure 7 of the post.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from .metrics import normalized_edit_distance


@dataclass
class Block:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    furniture: bool = False  # running header, footer, page number

    @classmethod
    def xywh(cls, name, x, y, w, h, furniture=False):
        return cls(name, x, y, x + w, y + h, furniture)


# Figure 7 of the post: a 360 x 450 two-column page with twelve blocks (x, y, width, height).
FIG7_SIZE = (360, 450)
FIG7_BLOCKS = [
    Block.xywh("H", 20, 12, 320, 18, True), Block.xywh("T", 40, 36, 280, 28), Block.xywh("Ab", 20, 70, 320, 52),
    Block.xywh("A", 20, 130, 155, 80), Block.xywh("F", 20, 218, 155, 70), Block.xywh("Fc", 20, 294, 155, 22),
    Block.xywh("Bp", 20, 322, 155, 86), Block.xywh("C", 185, 130, 155, 62), Block.xywh("Tc", 185, 200, 155, 20),
    Block.xywh("Tb", 185, 226, 155, 70), Block.xywh("D", 185, 304, 155, 72), Block.xywh("Fn", 185, 384, 155, 24),
    Block.xywh("P", 160, 418, 40, 20, True),
]
FIG7_REFERENCE = ["T", "Ab", "A", "F", "Fc", "Bp", "C", "Tc", "Tb", "D", "Fn"]


def order_score(pred: list[str], ref: list[str]) -> float:
    """Normalized edit distance between two block sequences. 0 means the reference order."""
    return normalized_edit_distance(pred, ref) if pred else 1.0


def raster_order(blocks: list[Block]) -> list[str]:
    """What a naive line scan does: sort by top edge, then by left edge."""
    return [b.name for b in sorted(blocks, key=lambda b: (b.y0, b.x0))]


def _widest_gap(blocks: list[Block], axis: int):
    """The widest empty band along one axis (0 = x, 1 = y): (width, where it starts), or (0, None) if none."""
    lo = (lambda b: b.x0) if axis == 0 else (lambda b: b.y0)
    hi = (lambda b: b.x1) if axis == 0 else (lambda b: b.y1)
    best, at, reach = 0.0, None, None
    for b in sorted(blocks, key=lo):
        if reach is not None and lo(b) - reach > best:
            best, at = lo(b) - reach, reach
        reach = hi(b) if reach is None else max(reach, hi(b))
    return best, at


def xy_cut(blocks: list[Block], min_gap: float = 1.0) -> list[str]:
    """Recursive XY-cut. Find the widest empty band across the blocks, horizontal or vertical, and cut there:
    top part before bottom part, left part before right part. Recurse on both parts.
    A group that cannot be cut either way falls back to top-edge order."""
    if len(blocks) <= 1:
        return [b.name for b in blocks]
    (gy, at_y), (gx, at_x) = _widest_gap(blocks, 1), _widest_gap(blocks, 0)
    if max(gx, gy) < min_gap:
        return raster_order(blocks)
    if gy >= gx:
        first, second = [b for b in blocks if b.y1 <= at_y], [b for b in blocks if b.y1 > at_y]
    else:
        first, second = [b for b in blocks if b.x1 <= at_x], [b for b in blocks if b.x1 > at_x]
    return xy_cut(first, min_gap) + xy_cut(second, min_gap)


def drop_margin_furniture(blocks: list[Block], page_h: float, band: float = 0.08) -> list[Block]:
    """Treat blocks that sit entirely in the top or bottom band of the page as furniture and leave them out."""
    return [b for b in blocks if not (b.y1 <= band * page_h or b.y0 >= (1 - band) * page_h)]


# ---- Synthetic pages with a known reading order -------------------------------------------------------------

LAYOUTS = ["1-col", "2-col", "3-col", "2-col + full-width figure", "3-col + figure over 2 cols",
           "2-col, boxes cross the gutter"]


def synthetic_page(layout: str, rng: random.Random, w: float = 1000, h: float = 1400):
    """A page of blocks and its reference order (furniture excluded). Columns are read top to bottom, left to right.
    A full-width figure splits the page into an upper and a lower band. A figure over two of three columns sits inside
    the left two columns, which are read as one band above it, then it, then the band below; the third column follows.
    In the last layout a few boxes are widened past the middle of the gutter, as loose detector boxes are."""
    blocks: list[Block] = []
    ref: list[str] = []
    k = 0

    def add(x0, y0, x1, y1, furniture=False):
        nonlocal k
        name = f"b{k}"
        k += 1
        blocks.append(Block(name, x0, y0, x1, y1, furniture))
        if not furniture:
            ref.append(name)
        return name

    m, gut = 60, rng.uniform(20, 40)
    add(m, 25, w - m, 25 + rng.uniform(20, 35), furniture=True)  # running header
    y = 130.0
    add(w * 0.2, y, w * 0.8, y + rng.uniform(40, 70))  # title
    y = blocks[-1].y1 + rng.uniform(15, 30)
    n_cols = int(layout[0])

    def column_stack(x0, x1, top, bottom):
        yy = top
        while True:
            hgt = rng.uniform(50, 220)
            if yy + hgt > bottom:
                break
            add(x0 + rng.uniform(0, 6), yy, x1 - rng.uniform(0, 6), yy + hgt)
            yy += hgt + rng.uniform(8, 25)

    col_w = (w - 2 * m - (n_cols - 1) * gut) / n_cols
    xs = [(m + i * (col_w + gut), m + i * (col_w + gut) + col_w) for i in range(n_cols)]
    bottom = h - 120
    if layout == "2-col + full-width figure":
        mid = rng.uniform(0.35, 0.6) * h
        for x0, x1 in xs:
            column_stack(x0, x1, y, mid - 20)
        add(m, mid, w - m, mid + rng.uniform(150, 300))
        top2 = blocks[-1].y1 + 25
        for x0, x1 in xs:
            column_stack(x0, x1, top2, bottom)
    elif layout == "3-col + figure over 2 cols":
        mid = rng.uniform(0.35, 0.55) * h
        for x0, x1 in xs[:2]:
            column_stack(x0, x1, y, mid - 20)
        add(xs[0][0], mid, xs[1][1], mid + rng.uniform(150, 280))
        top2 = blocks[-1].y1 + 25
        for x0, x1 in xs[:2]:
            column_stack(x0, x1, top2, bottom)
        column_stack(xs[2][0], xs[2][1], y, bottom)
    else:
        n_before = len(blocks)
        for x0, x1 in xs:
            column_stack(x0, x1, y, bottom)
        if layout == "2-col, boxes cross the gutter":
            # A detector's boxes are loose: widen a few body boxes so they reach into the gutter past its middle.
            for b in rng.sample(blocks[n_before:], k=min(3, len(blocks) - n_before)):
                b.x0, b.x1 = max(0.0, b.x0 - 0.6 * gut), min(w, b.x1 + 0.6 * gut)
    add(w / 2 - 20, h - 60, w / 2 + 20, h - 40, furniture=True)  # page number
    return blocks, ref


# ---- A small learned orderer, in the spirit of LayoutReader: learn "does A come before B" from boxes ----------

def pair_features(a: Block, b: Block, w: float, h: float) -> list[float]:
    fa = [a.x0 / w, a.y0 / h, a.x1 / w, a.y1 / h]
    fb = [b.x0 / w, b.y0 / h, b.x1 / w, b.y1 / h]
    return fa + fb + [fb[i] - fa[i] for i in range(4)]


class PairOrderer(nn.Module):
    """An MLP that scores P(a is read before b) from the two boxes. Blocks are ranked by how many others they precede."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def order(self, blocks: list[Block], w: float, h: float) -> list[str]:
        if len(blocks) < 2:
            return [b.name for b in blocks]
        pairs = [(i, j) for i in range(len(blocks)) for j in range(len(blocks)) if i != j]
        x = torch.tensor([pair_features(blocks[i], blocks[j], w, h) for i, j in pairs])
        p = torch.sigmoid(self(x))
        wins = [0.0] * len(blocks)
        for (i, _), pij in zip(pairs, p.tolist()):
            wins[i] += pij
        return [blocks[i].name for i in sorted(range(len(blocks)), key=lambda i: -wins[i])]


def pair_dataset(pages, w: float = 1000, h: float = 1400):
    xs, ys = [], []
    for blocks, ref in pages:
        body = [b for b in blocks if not b.furniture]
        rank = {n: i for i, n in enumerate(ref)}
        for a in body:
            for b in body:
                if a is not b:
                    xs.append(pair_features(a, b, w, h))
                    ys.append(float(rank[a.name] < rank[b.name]))
    return torch.tensor(xs), torch.tensor(ys)


def train_orderer(pages, steps: int = 1500, seed: int = 0, lr: float = 3e-3) -> PairOrderer:
    torch.manual_seed(seed)
    x, y = pair_dataset(pages)
    model = PairOrderer()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (512,), generator=g)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
