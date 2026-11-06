"""Synthetic street photos for the post's running example.

Each scene is a 4032 x 3024 "phone photo" of a street. A bakery has a large sign that is readable at any resolution.
A small plaque gives the opening hours, with digits 32 px tall in the original. A bus-stop sign with a time on it is
the distractor from the section "Searching over visual evidence". The plaque always sits inside one cell of a 4 x 4
grid, so a crop of that cell (1008 x 756) contains it whole.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

W, H = 4032, 3024
GRID = 4  # the plaque lies inside one cell of a GRID x GRID split
TEXT_H = 32  # digit height of the plaque text in original pixels
OPEN_TIMES = ["7:00", "7:30", "8:00", "8:30", "9:00", "10:00"]
CLOSE_TIMES = ["16:00", "16:30", "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "21:00"]
PRIOR_CLOSE = "18:00"  # the "bakeries usually close at six" prior from the post


@dataclass
class Scene:
    image: Image.Image
    open_time: str
    close_time: str
    plaque_box: tuple[int, int, int, int]  # x0, y0, x1, y1 in original pixels
    plaque_cell: tuple[int, int]  # (col, row) in the GRID x GRID split
    bus_time: str
    bus_box: tuple[int, int, int, int]
    seed: int
    facts: str = field(default="")  # oracle description: states the facts, not the answer


def font_for_digit_height(px: float) -> ImageFont.FreeTypeFont:
    """Pillow's bundled font, sized so that digits are `px` pixels tall."""
    probe = Image.new("L", (900, 200), 0)
    ImageDraw.Draw(probe).text((10, 10), "0123456789", font=ImageFont.load_default(size=100), fill=255)
    _, y0, _, y1 = probe.getbbox()  # ink rows, not the font's line box
    return ImageFont.load_default(size=max(4, round(100 * px / (y1 - y0))))


def cell_box(col: int, row: int, grid: int = GRID) -> tuple[int, int, int, int]:
    cw, ch = W // grid, H // grid
    return col * cw, row * ch, (col + 1) * cw, (row + 1) * ch


def _sign(draw, xy, text, font, fg, bg, pad):
    x, y = xy
    x0, y0, x1, y1 = draw.textbbox((x, y), text, font=font)
    box = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    draw.rectangle(box, fill=bg, outline=(40, 40, 40), width=max(2, pad // 4))
    draw.text((x, y), text, font=font, fill=fg)
    return tuple(int(v) for v in box)


def make_scene(seed: int, cell: tuple[int, int] | None = (GRID - 1, 0), distractor: bool = True) -> Scene:
    """Render one scene. `cell=None` puts the plaque in a random cell; the default is the top-right cell.

    `distractor=False` leaves out the bus sign and draws everything else identically (same seed, same photo).
    """
    rng = random.Random(seed)
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    # Sky, then a row of buildings, then pavement and road.
    for y in range(0, 900, 6):
        c = 150 + y * 80 // 900
        d.rectangle((0, y, W, y + 6), fill=(c - 40, c, 235))
    x = 0
    while x < W:
        bw = rng.randint(500, 1100)
        top = rng.randint(150, 650)
        col = tuple(rng.randint(90, 200) for _ in range(3))
        d.rectangle((x, top, x + bw, 2300), fill=col)
        for wx in range(x + 60, x + bw - 120, 200):
            for wy in range(top + 80, 1500, 260):
                d.rectangle((wx, wy, wx + 110, wy + 160), fill=(60, 80, 110), outline=(230, 230, 230), width=6)
        x += bw
    d.rectangle((0, 2300, W, 2550), fill=(170, 170, 165))
    d.rectangle((0, 2550, W, H), fill=(70, 70, 75))
    for lx in range(100, W, 600):
        d.rectangle((lx, 2770, lx + 300, 2800), fill=(235, 235, 235))
    # Parked cars and trees, as clutter.
    for _ in range(rng.randint(2, 4)):
        cx = rng.randint(0, W - 900)
        cc = tuple(rng.randint(20, 230) for _ in range(3))
        d.rounded_rectangle((cx, 2380, cx + 800, 2650), 60, fill=cc)
        for wx in (cx + 140, cx + 560):
            d.ellipse((wx, 2580, wx + 140, 2720), fill=(20, 20, 20))
    for _ in range(rng.randint(1, 3)):
        tx = rng.randint(0, W - 400)
        d.rectangle((tx + 170, 1900, tx + 230, 2350), fill=(90, 60, 30))
        d.ellipse((tx, 1500, tx + 400, 2000), fill=(40, rng.randint(110, 160), 50))

    # Bakery front: big sign, door, shop window. Readable even at 448 px wide.
    bx = rng.randint(1100, 1700)
    d.rectangle((bx, 1500, bx + 1300, 2300), fill=(150, 90, 60))
    _sign(d, (bx + 160, 1560), "BAKERY", font_for_digit_height(150), (250, 240, 210), (90, 40, 30), 30)
    d.rectangle((bx + 120, 1850, bx + 560, 2300), fill=(60, 35, 25))
    d.rectangle((bx + 680, 1850, bx + 1220, 2150), fill=(200, 220, 230), outline=(60, 35, 25), width=12)
    # The plaque with the opening hours, inside the chosen cell.
    if cell is None:
        cell = (rng.randrange(GRID), rng.randrange(GRID))
    open_t, close_t = rng.choice(OPEN_TIMES), rng.choice(CLOSE_TIMES)
    text = f"OPEN {open_t}-{close_t}"
    font = font_for_digit_height(TEXT_H)
    cx0, cy0, cx1, cy1 = cell_box(*cell)
    tw, th = d.textbbox((0, 0), text, font=font)[2:]
    px = rng.randint(cx0 + 40, cx1 - tw - 40)
    py = rng.randint(cy0 + 40, cy1 - th - 40)
    plaque = _sign(d, (px, py), text, font, (245, 245, 235), (30, 45, 40), 14)

    # The distractor: a bus-stop sign with a time, in a different cell, at the same text size.
    bus_t = rng.choice([t for t in CLOSE_TIMES if t != close_t])
    bus_text = f"BUS 12 - {bus_t}"
    cells = [(c, r) for c in range(GRID) for r in range(GRID) if (c, r) != cell]
    bc = rng.choice(cells)
    bx0, by0, bx1, by1 = cell_box(*bc)
    tw, th = d.textbbox((0, 0), bus_text, font=font)[2:]
    sx = rng.randint(bx0 + 40, bx1 - tw - 40)
    sy = rng.randint(by0 + 40, by1 - th - 40)
    if distractor:
        bus = _sign(d, (sx, sy), bus_text, font, (20, 20, 20), (240, 240, 240), 14)
    else:
        bus = (-1, -1, -1, -1)  # no sign: no view contains this box

    facts = f"A small plaque in the photo reads 'OPEN {open_t}-{close_t}'. "
    if distractor:
        facts += f"A bus-stop sign reads 'BUS 12 - {bus_t}'. "
    facts += "A large sign reads 'BAKERY'."
    return Scene(img, open_t, close_t, plaque, cell, bus_t, bus, seed, facts)


def crop(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return img.crop(box)


def resize_to(img: Image.Image, w: int, h: int) -> Image.Image:
    """The resolution policy: an area-averaging resize. This is the low-pass filter from the post."""
    return img.resize((w, h), Image.BOX)
