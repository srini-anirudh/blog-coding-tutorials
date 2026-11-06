"""The harness around a computer-use model: coordinates, validation, a reversibility gate, and context cost.

The model chooses actions. This code turns them into real events and refuses the ones that should not run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from urllib.parse import urlparse


# -- screenshot capture and coordinate mapping ---------------------------------------------------------------------
def fit_to_budget(w: int, h: int, max_tokens: int, patch: int = 14) -> tuple[int, int, float]:
    """Shrink a w x h screen (never enlarge it), keeping its aspect ratio, so it fits max_tokens patches.
    Returns the new size and the one scale factor the harness must keep to map clicks back."""
    s = min(1.0, math.sqrt(max_tokens / ((w / patch) * (h / patch))))
    return int(w * s), int(h * s), s


def to_screen(point, fmt: str, screen_w: int, screen_h: int, scale: float = 1.0) -> tuple[int, int]:
    """Map a model's coordinate output to real screen pixels.

    norm01      x, y in [0, 1], as SeeClick writes them
    bin999      x, y in 000 to 999, as CogAgent writes box corners
    resized_px  pixels in the resized screenshot the model saw (needs the capture scale factor)
    screen_px   raw screen pixels, as UGround writes them
    """
    x, y = point
    if fmt == "norm01":
        x, y = x * screen_w, y * screen_h
    elif fmt == "bin999":
        x, y = x / 999 * screen_w, y / 999 * screen_h
    elif fmt == "resized_px":
        x, y = x / scale, y / scale
    elif fmt != "screen_px":
        raise ValueError(f"unknown coordinate format {fmt}")
    return int(round(x)), int(round(y))


# -- action validation -----------------------------------------------------------------------------------------------
class Rejected(Exception):
    pass


def validate(action: tuple, screen_w: int, screen_h: int, element_ids=(), allowed=("click", "type", "element",
             "api")) -> tuple:
    """Refuse malformed actions before they run: unknown kinds, points off the screen, ids that do not exist."""
    kind = action[0]
    if kind not in allowed:
        raise Rejected(f"action {kind!r} not allowed")
    if kind == "click":
        x, y = action[1], action[2]
        if not (0 <= x < screen_w and 0 <= y < screen_h):
            raise Rejected(f"click ({x}, {y}) is off the {screen_w}x{screen_h} screen")
    if kind == "element" and action[1] not in element_ids:
        raise Rejected(f"element {action[1]!r} is not on screen")
    return action


# -- reversibility and confirmation gate -----------------------------------------------------------------------------
IRREVERSIBLE = {"send", "delete", "pay", "publish", "export", "submit"}


@dataclass
class Gate:
    """Classify each proposed action by reversibility. Irreversible actions, and anything that sends data to a domain
    outside the allow-list, wait for a human. The model cannot argue its way past this: it is code, not a prompt."""

    allowed_domains: set = field(default_factory=lambda: {"expenses.example"})
    confirm: callable = staticmethod(lambda action: False)  # the human; denies by default
    audit: list = field(default_factory=list)

    def risk(self, action: dict) -> str:
        target = action.get("to", "")
        domain = target.split("@")[-1] if "@" in target else urlparse(target).netloc or target
        if domain and domain not in self.allowed_domains:
            return "external"
        return "irreversible" if action["verb"] in IRREVERSIBLE else "safe"

    def check(self, action: dict) -> bool:
        level = self.risk(action)
        ok = level == "safe" or bool(self.confirm(action))
        self.audit.append((action["verb"], level, "run" if ok else "blocked"))
        return ok


# -- observation compression -----------------------------------------------------------------------------------------
def context_tokens(step: int, image_tokens: int, keep_images: int | None, text_per_step: int = 30,
                   task_tokens: int = 50) -> int:
    """Prompt size at a given step. keep_images=None keeps every past screenshot; otherwise only the most recent
    keep_images stay as images and older steps become a short text line (action and what changed)."""
    n_img = step + 1 if keep_images is None else min(step + 1, keep_images)
    n_text = step + 1 - n_img
    return task_tokens + n_img * image_tokens + n_text * text_per_step + step * 10  # 10 tokens per action
