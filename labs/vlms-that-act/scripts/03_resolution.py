"""Pixels per visual token, p = sqrt(W * H / N), and how many 24-pixel icons share one token.

Post section: "Resolution decides what can be pointed at" (Figure 6). The arithmetic is exact.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actlab.resolution import distinguishable_icons, icon_strip, icons_per_token, pixels_per_token  # noqa: E402
from actlab.report import Report  # noqa: E402

rep = Report("03_resolution")
rep(f"{'screen':>11s} {'tokens':>7s} {'px/token':>9s} {'24px icons per token width':>27s}")
table = []
for w, h in ((1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)):
    for n in (256, 1024, 4096):
        p = pixels_per_token(w, h, n)
        table.append(dict(w=w, h=h, tokens=n, px_per_token=p))
        rep(f"{w:>5d}x{h:<5d} {n:7d} {p:9.1f} {p / 24:27.2f}")
rep("")
_, boxes = icon_strip(n_icons=16, icon=24)
rep("A strip of 16 adjacent 24-pixel icons. How many icons own a token column alone?")
for w, h, n in ((1920, 1080, 4096), (1920, 1080, 1024), (3840, 2160, 1024), (3840, 2160, 256)):
    p = pixels_per_token(w, h, n)
    counts = icons_per_token(boxes, p)
    rep(f"  {w}x{h}, {n:4d} tokens: p = {p:5.1f}px, icons per token column = {counts}, "
        f"alone: {distinguishable_icons(boxes, p)}/16")
rep.save({"table": table})
