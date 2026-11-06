"""The intake bench: what each resolution strategy does to an image before the encoder sees it.

Every strategy is split in two. `plan_*` works from the image size alone and returns the geometry and the
token count. `render` applies a plan to a real picture. The token counts follow the public processors:

    clip_crop      CLIP: short side to 224, centre crop, ViT-L/14                 -> 256 patches
    squash         resize to 336 x 336, ViT-L/14                                  -> 576 patches
    letterbox      long side to 384, pad to square, SigLIP SO400M/14              -> 729 patches
    anyres         LLaVA-NeXT: best grid of 336 tiles + a squashed 336 base image -> 576 per view
    internvl       InternVL: closest-aspect grid of 1-12 tiles of 448 + thumbnail -> 1024 patches, 256 tokens per view
    qwen           Qwen2-VL: snap to multiples of 28 inside [min_pixels, max_pixels] -> (h/14)(w/14) patches, /4 tokens
    naflex         SigLIP 2 NaFlex: largest patch-16 grid with at most 1024 patches
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

# Qwen2-VL / Qwen2.5-VL preprocessor_config.json on the Hub. The transformers code default is much lower.
QWEN_MIN_PIXELS = 3136
QWEN_MAX_PIXELS = 12845056
QWEN_CODE_DEFAULT_MAX_PIXELS = 28 * 28 * 1280
LLAVA_NEXT_PINPOINTS = [(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]  # (height, width)


@dataclass
class Plan:
    """canvas = resize(image, resize_to) pasted at (ox, oy) on a canvas of canvas_size. Tiles cut the canvas."""

    name: str
    resize_to: tuple[int, int]  # (width, height) of the resized image
    canvas_size: tuple[int, int]  # (width, height) of the canvas the tiles are cut from
    offset: tuple[int, int] = (0, 0)  # where the resized image lands on the canvas (negative = cropped)
    patches: int = 0  # encoder patch tokens, all views
    llm_tokens: int = 0  # tokens that reach the language model
    views: int = 1  # tiles plus thumbnail
    fill: tuple[int, int, int] = (0, 0, 0)
    notes: list[str] = field(default_factory=list)

    def scale(self, width: int, height: int) -> tuple[float, float]:
        return self.resize_to[0] / width, self.resize_to[1] / height

    def distortion(self, width: int, height: int) -> float:
        """How much the aspect ratio was stretched: 0 means preserved."""
        sx, sy = self.scale(width, height)
        return max(sx, sy) / min(sx, sy) - 1

    def visible(self) -> tuple[int, int, int, int]:
        """The part of the canvas covered by image pixels: (x0, y0, x1, y1)."""
        ox, oy = self.offset
        rw, rh = self.resize_to
        cw, ch = self.canvas_size
        return max(ox, 0), max(oy, 0), min(ox + rw, cw), min(oy + rh, ch)

    def discarded(self) -> float:
        """Fraction of the source image cropped away."""
        x0, y0, x1, y1 = self.visible()
        return 1 - (x1 - x0) * (y1 - y0) / (self.resize_to[0] * self.resize_to[1])

    def padding(self) -> float:
        """Fraction of the canvas that is padding, not image."""
        x0, y0, x1, y1 = self.visible()
        return 1 - (x1 - x0) * (y1 - y0) / (self.canvas_size[0] * self.canvas_size[1])


def plan_clip_crop(width: int, height: int, size: int = 224, patch: int = 14) -> Plan:
    s = size / min(width, height)
    rw, rh = round(width * s), round(height * s)
    ox, oy = -int(round((rw - size) / 2)), -int(round((rh - size) / 2))
    n = (size // patch) ** 2
    return Plan("CLIP crop 224", (rw, rh), (size, size), (ox, oy), n, n)


def plan_squash(width: int, height: int, size: int = 336, patch: int = 14) -> Plan:
    n = (size // patch) ** 2
    return Plan("Squash 336", (size, size), (size, size), (0, 0), n, n)


def plan_letterbox(width: int, height: int, size: int = 384, patch: int = 14) -> Plan:
    s = size / max(width, height)
    rw, rh = max(1, round(width * s)), max(1, round(height * s))
    n = (size // patch) ** 2
    return Plan("Letterbox 384", (rw, rh), (size, size), ((size - rw) // 2, (size - rh) // 2), n, n, fill=(128, 128, 128))


def select_best_resolution(original_size: tuple[int, int], possible: list[tuple[int, int]]) -> tuple[int, int]:
    """LLaVA-NeXT: the grid that keeps the most image pixels, then wastes the fewest. Sizes are (height, width)."""
    oh, ow = original_size
    best, best_eff, best_waste = None, 0, float("inf")
    for h, w in possible:
        s = min(w / ow, h / oh)
        eff = min(int(ow * s) * int(oh * s), ow * oh)
        waste = w * h - eff
        if eff > best_eff or (eff == best_eff and waste < best_waste):
            best, best_eff, best_waste = (h, w), eff, waste
    return best


def plan_anyres(width: int, height: int, tile: int = 336, patch: int = 14) -> Plan:
    th, tw = select_best_resolution((height, width), LLAVA_NEXT_PINPOINTS)
    sw, sh = tw / width, th / height
    if sw < sh:
        rw, rh = tw, min(math.ceil(height * sw), th)
    else:
        rw, rh = min(math.ceil(width * sh), tw), th
    tiles = (tw // tile) * (th // tile)
    per_view = (tile // patch) ** 2
    plan = Plan("AnyRes tiles", (rw, rh), (tw, th), ((tw - rw) // 2, (th - rh) // 2), (tiles + 1) * per_view,
                (tiles + 1) * per_view, tiles + 1)
    plan.notes.append("token count before LLaVA-NeXT removes padded features and adds row-end tokens")
    return plan


def tile_grids(min_tiles: int, max_tiles: int) -> list[tuple[int, int]]:
    """Every (columns, rows) grid with min_tiles <= columns * rows <= max_tiles, fewest tiles first."""
    grids = [(c, r) for c in range(1, max_tiles + 1) for r in range(1, max_tiles + 1) if min_tiles <= c * r <= max_tiles]
    return sorted(grids, key=lambda g: g[0] * g[1])


def closest_grid(width: int, height: int, tile: int = 448, min_tiles: int = 1, max_tiles: int = 12) -> tuple[int, int]:
    """InternVL's find_closest_aspect_ratio. On a tie, take the larger grid while the image is big enough."""
    aspect, area = width / height, width * height
    best, best_diff = (1, 1), float("inf")
    for c, r in tile_grids(min_tiles, max_tiles):
        diff = abs(aspect - c / r)
        if diff < best_diff:
            best, best_diff = (c, r), diff
        elif diff == best_diff and area > 0.5 * tile * tile * c * r:
            best = (c, r)
    return best


def plan_internvl(width: int, height: int, tile: int = 448, patch: int = 14, shuffle: int = 2, max_tiles: int = 12) -> Plan:
    c, r = closest_grid(width, height, tile, 1, max_tiles)
    views = c * r + (1 if c * r > 1 else 0)  # a thumbnail is added whenever there is more than one tile
    per_view = (tile // patch) ** 2
    return Plan("InternVL tiles + shuffle", (c * tile, r * tile), (c * tile, r * tile), (0, 0), views * per_view,
                views * per_view // shuffle**2, views)


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = QWEN_MIN_PIXELS,
                 max_pixels: int = QWEN_MAX_PIXELS) -> tuple[int, int]:
    """Qwen2-VL: round both sides to multiples of `factor`, then rescale into [min_pixels, max_pixels]."""
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w


def plan_qwen(width: int, height: int, max_pixels: int = QWEN_MAX_PIXELS, patch: int = 14, merge: int = 2) -> Plan:
    h, w = smart_resize(height, width, patch * merge, QWEN_MIN_PIXELS, max_pixels)
    n = (h // patch) * (w // patch)
    return Plan("Native (Qwen)", (w, h), (w, h), (0, 0), n, n // merge**2)


def naflex_size(height: int, width: int, patch: int = 16, max_patches: int = 1024, eps: float = 1e-5) -> tuple[int, int]:
    """SigLIP 2 NaFlex: binary-search the largest scale whose patch-multiple grid fits max_patches."""

    def side(scale: float, size: int) -> int:
        return int(max(patch, math.ceil(size * scale / patch) * patch))

    lo, hi = eps / 10, 100.0
    while hi - lo >= eps:
        mid = (lo + hi) / 2
        if (side(mid, height) / patch) * (side(mid, width) / patch) <= max_patches:
            lo = mid
        else:
            hi = mid
    return side(lo, height), side(lo, width)


def plan_naflex(width: int, height: int, patch: int = 16, max_patches: int = 1024) -> Plan:
    h, w = naflex_size(height, width, patch, max_patches)
    n = (h // patch) * (w // patch)
    return Plan("NaFlex 1024", (w, h), (w, h), (0, 0), n, n)


STRATEGIES = {
    "clip_crop": plan_clip_crop,
    "squash": plan_squash,
    "letterbox": plan_letterbox,
    "anyres": plan_anyres,
    "internvl": plan_internvl,
    "qwen": plan_qwen,
    "naflex": plan_naflex,
}

# The five shapes on the post's bench, long side 1024 px. (width, height)
SHAPES = {
    "A4 page 1:1.41": (726, 1024),
    "Photo 4:3": (1024, 768),
    "Screenshot 16:9": (1024, 576),
    "Receipt 1:3.2": (320, 1024),
    "Panorama 3:1": (1024, 341),
}
STROKES = (24, 15, 9, 5, 3)  # bar widths in the master image, as on the post's bench


def render(image: Image.Image, plan: Plan) -> Image.Image:
    """Apply a plan: bicubic resize (PIL filters with antialiasing when shrinking), then crop or pad."""
    resized = image.resize(plan.resize_to, Image.Resampling.BICUBIC)
    canvas = Image.new(image.mode, plan.canvas_size, plan.fill if image.mode == "RGB" else plan.fill[0])
    canvas.paste(resized, plan.offset)
    return canvas


@dataclass
class Grating:
    stroke: int
    vertical: bool  # vertical bars test horizontal sampling
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in the master image


def resolution_target(width: int, height: int, shift: int = 0) -> tuple[Image.Image, list[Grating]]:
    """A white page with five bar groups along its long axis. Each group has 3 dark bars, both orientations.

    `shift` moves every grating by a few master pixels, so the bars land at a different phase of the output grid.
    """
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    long_is_x = width >= height
    long_side, short_side = (width, height) if long_is_x else (height, width)
    gratings = []
    for i, w in enumerate(STROKES):
        span = 5 * w
        t = (i + 0.5) * long_side / len(STROKES)
        for j, vertical in enumerate((True, False)):
            u = short_side * (0.3 if j == 0 else 0.7)
            cx, cy = (t, u) if long_is_x else (u, t)
            x0, y0 = int(round(cx - span / 2)) + shift, int(round(cy - span / 2)) + shift
            for k in range(3):
                a = 2 * k * w
                rect = (x0 + a, y0, x0 + a + w - 1, y0 + span - 1) if vertical else (x0, y0 + a, x0 + span - 1, y0 + a + w - 1)
                draw.rectangle(rect, fill=0)
            gratings.append(Grating(w, vertical, (x0, y0, x0 + span, y0 + span)))
    return img, gratings


def grating_contrast(canvas: Image.Image, plan: Plan, g: Grating, src_size: tuple[int, int]) -> float | None:
    """How much of the black-to-white swing survives intake: max - min of the profile across the bars,
    between the first and last bar centres. 1 means crisp bars, 0 a grey smear.

    Returns None when the grating was cropped away.
    """
    sx, sy = plan.scale(*src_size)
    ox, oy = plan.offset
    x0, y0, x1, y1 = g.box
    box = (x0 * sx + ox, y0 * sy + oy, x1 * sx + ox, y1 * sy + oy)
    cw, ch = canvas.size
    inside = (min(box[2], cw) - max(box[0], 0)) * (min(box[3], ch) - max(box[1], 0))
    if min(box[2], cw) <= max(box[0], 0) or min(box[3], ch) <= max(box[1], 0) or inside < 0.9 * (box[2] - box[0]) * (box[3] - box[1]):
        return None
    a = np.asarray(canvas, dtype=np.float64) / 255.0
    s = sx if g.vertical else sy
    lo, hi = (box[0], box[2]) if g.vertical else (box[1], box[3])
    first, last = lo + 0.5 * g.stroke * s, hi - 0.5 * g.stroke * s  # centres of bar 1 and bar 3
    # Average along the bars (middle 60 % of their length) to get a 1-D profile across them.
    plo, phi = (box[1], box[3]) if g.vertical else (box[0], box[2])
    m0, m1 = int(plo + 0.2 * (phi - plo)), max(int(plo + 0.2 * (phi - plo)) + 1, int(math.ceil(plo + 0.8 * (phi - plo))))
    profile = a[m0:m1, :].mean(axis=0) if g.vertical else a[:, m0:m1].mean(axis=1)
    i0, i1 = int(math.floor(first)), int(math.ceil(last)) + 1
    seg = profile[max(i0, 0): min(i1, len(profile))]
    if len(seg) < 2:
        return 0.0
    return float(seg.max() - seg.min())


def bench(shape: str, strategy: str, threshold: float = 0.5, shifts: tuple[int, ...] = (0, 1, 2, 3)) -> dict:
    """Run one cell of the intake bench and return what the encoder actually receives.

    A bar group counts as resolved when at least `threshold` of the black-to-white swing survives in both
    orientations, at every phase in `shifts` (the worst phase decides).
    """
    width, height = SHAPES[shape]
    plan = STRATEGIES[strategy](width, height)
    sx, sy = plan.scale(width, height)
    contrasts: dict[int, list] = {}
    for shift in shifts:
        page, gratings = resolution_target(width, height, shift)
        canvas = render(page, plan)
        for g in gratings:
            contrasts.setdefault(g.stroke, []).append(grating_contrast(canvas, plan, g, (width, height)))
    worst = {w: (None if any(c is None for c in cs) else min(cs)) for w, cs in contrasts.items()}
    resolved_measured = sum(1 for c in worst.values() if c is not None and c >= threshold)
    # The post's proxy: a bar is resolved when its stroke survives at >= 1.6 output pixels in both directions.
    resolved_proxy = sum(1 for w, c in worst.items() if c is not None and w * min(sx, sy) >= 1.6)
    return {
        "shape": shape, "strategy": plan.name, "patches": plan.patches, "llm_tokens": plan.llm_tokens,
        "views": plan.views, "scale_x": round(sx, 4), "scale_y": round(sy, 4),
        "distortion_pct": round(100 * plan.distortion(width, height), 1),
        "discarded_pct": round(100 * plan.discarded(), 1), "padding_pct": round(100 * plan.padding(), 1),
        "resolved_measured": resolved_measured, "resolved_proxy": resolved_proxy,
        "worst_swing": {w: None if c is None else round(c, 3) for w, c in worst.items()},
        "notes": plan.notes,
    }


def vit_flops(n: int, dim: int, mlp: int, depth: int, window: int | None = None, full_layers: int = 0,
              gated: bool = True) -> dict:
    """Forward FLOPs of a ViT on n tokens. Attention is 4 n^2 d per layer (QK^T and PV), or 4 n w d with windows of w."""
    proj = 2 * n * (4 * dim * dim + (3 if gated else 2) * dim * mlp)
    full = 4 * n * n * dim
    win = 4 * n * min(window or n, n) * dim
    attn = full_layers * full + (depth - full_layers) * (win if window else full)
    return {"projections": depth * proj, "attention": attn, "total": depth * proj + attn}
