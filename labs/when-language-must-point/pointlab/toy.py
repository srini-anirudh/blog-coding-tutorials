"""A toy referring task, small enough to train on a CPU in minutes.

Each scene is a 64 x 64 image with three coloured shapes on a dark background. The query names one colour.
The answer is that shape's region. Shapes are compact (rectangle, ellipse) or irregular (curved band, thin
diagonal bar, pie wedge), so the toy has the same split the post draws between boxes and masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import annular_sector, ellipse, polygon_mask, rotated_bar, tight_box, wedge

SIZE = 64
COLOURS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0.35, 1), "yellow": (1, 1, 0),
           "magenta": (1, 0, 1), "cyan": (0, 1, 1)}
COLOUR_NAMES = list(COLOURS)
SHAPES = ["rect", "ellipse", "band", "bar", "wedge"]
IRREGULAR = {"band", "bar", "wedge"}


def _shape_polygon(kind: str, rng: np.random.Generator):
    """A polygon centred near the origin; the caller moves it into the canvas."""
    u = rng.uniform
    if kind == "rect":
        w, h = u(8, 26), u(8, 26)
        return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    if kind == "ellipse":
        return ellipse(0, 0, u(5, 13), u(5, 13), n=64)
    if kind == "band":
        r2 = u(15, 26)
        span = u(90, 170)
        a0 = u(0, 360)
        return annular_sector(0, 0, r2 - u(4, 6.5), r2, a0, a0 + span, n=24)
    if kind == "bar":
        ang = u(25, 65) * rng.choice([-1, 1])
        return rotated_bar(0, 0, u(28, 50), u(3, 5), ang)
    if kind == "wedge":
        r = u(16, 30)
        a0 = u(0, 360)
        return wedge(0, 0, r, a0, a0 + u(30, 75), n=16)
    raise ValueError(kind)


@dataclass
class Scene:
    image: np.ndarray            # (3, H, W) float32
    masks: list[np.ndarray]      # visible mask of each object, (H, W) bool
    colours: list[int]           # index into COLOUR_NAMES
    kinds: list[str]


def _touch(a: np.ndarray, b: np.ndarray) -> bool:
    """Do two masks overlap or share an edge (4-connectivity)?"""
    grown = a.copy()
    grown[1:] |= a[:-1]
    grown[:-1] |= a[1:]
    grown[:, 1:] |= a[:, :-1]
    grown[:, :-1] |= a[:, 1:]
    return bool((grown & b).any())


def make_scene(rng: np.random.Generator, n_objects: int = 3, noise: float = 0.05, separate: bool = True) -> Scene:
    """separate=True rejects scenes where shapes touch; separate=False lets them overlap and occlude."""
    while True:
        colours = rng.choice(len(COLOUR_NAMES), n_objects, replace=False).tolist()
        kinds = [SHAPES[k] for k in rng.integers(0, len(SHAPES), n_objects)]
        full = []
        for kind in kinds:
            pts = np.array(_shape_polygon(kind, rng))
            lo, hi = pts.min(0), pts.max(0)
            span = hi - lo
            off = np.array([rng.uniform(1 - lo[0], SIZE - 1 - span[0] - lo[0]),
                            rng.uniform(1 - lo[1], SIZE - 1 - span[1] - lo[1])])
            full.append(polygon_mask(pts + off, SIZE, SIZE))
        # Later shapes are painted on top, so each object keeps only its visible pixels.
        visible = []
        for i, m in enumerate(full):
            v = m.copy()
            for later in full[i + 1:]:
                v &= ~later
            visible.append(v)
        if min(v.sum() for v in visible) < 25:
            continue
        if separate and any(_touch(full[i], full[j]) for i in range(n_objects) for j in range(i + 1, n_objects)):
            continue
        break
    img = np.zeros((3, SIZE, SIZE), np.float32)
    for m, c in zip(visible, colours):
        img[:, m] = np.array(COLOURS[COLOUR_NAMES[c]], np.float32)[:, None]
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return Scene(img, visible, colours, kinds)


@dataclass
class Batch:
    image: torch.Tensor      # (B, 3, H, W)
    query: torch.Tensor      # (B,) colour index
    mask: torch.Tensor       # (B, H, W) float, target region
    box: torch.Tensor        # (B, 4) target tight box, pixels
    kind: list[str]
    other_boxes: list[list]  # boxes of the distractors, for the wrong-object rate
    touching: torch.Tensor   # (B,) bool, True when the scene's shapes may touch or overlap


def make_dataset(n: int, seed: int, touching_share: float = 0.5) -> Batch:
    """n scenes; a share of them may have touching or overlapping shapes, the rest are kept apart."""
    rng = np.random.default_rng(seed)
    imgs, qs, ms, bs, ks, others, tc = [], [], [], [], [], [], []
    for _ in range(n):
        touch = bool(rng.random() < touching_share)
        tc.append(touch)
        s = make_scene(rng, separate=not touch)
        t = int(rng.integers(0, len(s.colours)))
        imgs.append(s.image)
        qs.append(s.colours[t])
        ms.append(s.masks[t])
        bs.append(tight_box(s.masks[t]))
        ks.append(s.kinds[t])
        others.append([tight_box(m) for j, m in enumerate(s.masks) if j != t])
    return Batch(torch.from_numpy(np.stack(imgs)), torch.tensor(qs), torch.from_numpy(np.stack(ms)).float(),
                 torch.tensor(bs, dtype=torch.float32), ks, others, torch.tensor(tc))


def subset(b: Batch, idx) -> Batch:
    idx = torch.as_tensor(idx)
    il = idx.tolist()
    return Batch(b.image[idx], b.query[idx], b.mask[idx], b.box[idx], [b.kind[i] for i in il],
                 [b.other_boxes[i] for i in il], b.touching[idx])


def connected_components(fg: torch.Tensor) -> torch.Tensor:
    """Label 4-connected foreground components by repeated max-pooling of pixel ids. (B, H, W) bool -> ids, 0 = bg."""
    b, h, w = fg.shape
    ids = torch.arange(1, h * w + 1, dtype=torch.float32).view(1, h, w).expand(b, h, w).clone()
    ids = ids * fg
    for _ in range(h * w):
        p = torch.nn.functional.pad(ids, (1, 1, 1, 1))
        nb = torch.stack([p[:, 1:-1, 1:-1], p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]]).amax(0)
        new = nb * fg
        if torch.equal(new, ids):
            break
        ids = new
    return ids.long()


def proposals(image: torch.Tensor, thr: float = 0.3, min_area: int = 10) -> list[list[torch.Tensor]]:
    """A class-agnostic proposer: every connected blob of non-background pixels is one candidate region.
    Two touching shapes become one blob, so neither can be selected on its own."""
    fg = image.amax(1) > thr
    ids = connected_components(fg)
    out = []
    for i in range(ids.shape[0]):
        regions = []
        for v in ids[i].unique().tolist():
            if v == 0:
                continue
            m = ids[i] == v
            if m.sum() >= min_area:
                regions.append(m)
        out.append(regions)
    return out


def bins_to_box(k: torch.Tensor, bins: int = 1000) -> torch.Tensor:
    """0..bins-1 coordinate tokens -> pixel box at bin centres."""
    return (k.float() + 0.5) / bins * SIZE


def box_to_bins(box: torch.Tensor, bins: int = 1000) -> torch.Tensor:
    return (box / SIZE * bins).floor().clamp(0, bins - 1).long()


def batch_box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ix = (torch.minimum(a[:, 2], b[:, 2]) - torch.maximum(a[:, 0], b[:, 0])).clamp(min=0)
    iy = (torch.minimum(a[:, 3], b[:, 3]) - torch.maximum(a[:, 1], b[:, 1])).clamp(min=0)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)
    union = area(a) + area(b) - inter
    return torch.where(union > 0, inter / union.clamp(min=1e-9), torch.zeros_like(union))


def box_masks(box: torch.Tensor) -> torch.Tensor:
    """Pixel boxes (B, 4) -> (B, H, W) masks of pixel centres inside each box."""
    c = torch.arange(SIZE, dtype=torch.float32) + 0.5
    inx = (c[None, :] >= box[:, 0:1]) & (c[None, :] <= box[:, 2:3])
    iny = (c[None, :] >= box[:, 1:2]) & (c[None, :] <= box[:, 3:4])
    return (iny[:, :, None] & inx[:, None, :]).float()


def fill_ratios(b: Batch) -> torch.Tensor:
    area = (b.box[:, 2] - b.box[:, 0]) * (b.box[:, 3] - b.box[:, 1])
    return b.mask.sum((1, 2)) / area

