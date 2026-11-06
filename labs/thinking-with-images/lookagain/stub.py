"""A stub reader with the same interface as QwenVLM, for offline tests.

It reads a rendered view through `img.info` instead of pixels: text is legible only when its digit height in the
view is at least `legible_px`. Otherwise it falls back on the language prior, 18:00. This is the post's toy model of
perception, made executable, so search and voting logic can be tested without a 4 GB download.
"""
from __future__ import annotations

import re

from .answers import CAPTION, FIND, VERIFY, VERIFY_GENERIC
from .scene import PRIOR_CLOSE


class StubVLM:
    def __init__(self, legible_px: float = 9.0, prefer_bus: bool = False):
        self.legible_px = legible_px
        self.prefer_bus = prefer_bus  # True: read the bus sign first when both are legible (a distractor-prone reader)
        self.calls = 0

    def _read(self, images) -> tuple[str | None, str | None]:
        """(plaque closing time, bus time), each None if not legible in any image."""
        close = bus = None
        for im in images:
            v, s = im.info.get("view"), im.info.get("scene")
            if v is None or v.text_px < self.legible_px:
                continue
            if v.contains(s.plaque_box):
                close = f"{s.open_time}-{s.close_time}"
            if v.contains(s.bus_box):
                bus = s.bus_time
        return close, bus

    def _answer(self, images, prompt) -> str:
        if not images:
            m = re.search(r"OPEN \S+?-(\d+:\d+)", prompt)
            return m.group(1) if m else PRIOR_CLOSE
        close, bus = self._read(images)
        if prompt == CAPTION:
            return " ".join(t for t in (f"OPEN {close}" if close else None, f"BUS 12 - {bus}" if bus else None) if t)
        if bus and (self.prefer_bus or not close):
            return bus
        return close or PRIOR_CLOSE

    def generate(self, items, n=1, **kw):
        self.calls += len(items)
        return [[self._answer(ims, p)] * n for ims, p in items]

    def ask(self, images, prompt, **kw):
        return self.generate([(images, prompt)])[0][0]

    def yes_prob(self, items):
        self.calls += len(items)
        out = []
        for ims, p in items:
            s = ims[0].info["scene"]
            v = ims[0].info["view"]
            if p == FIND:
                out.append(0.9 if v.contains(s.plaque_box) else 0.6 if v.contains(s.bus_box) else 0.1)
            elif p in (VERIFY, VERIFY_GENERIC):
                close, bus = self._read(ims)
                out.append(0.9 if close else 0.2 if bus else 0.3)
            else:
                out.append(0.5)
        return out
