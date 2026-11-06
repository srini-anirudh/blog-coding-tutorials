"""MiniDesk: a tiny expense app that an agent operates through pixels.

The app has a hidden state (the form, the open menu, any dialog, and the records saved on the "server"). The agent
never sees that state. It sees a 96 x 96 screenshot rendered from it, or one of the structured views in
``observe.py``. The checker grades a task by reading the saved records, not the picture.

Actions are the pixel-level ones from the post: ``("click", x, y)`` and ``("type", text)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

SIZE = 96  # logical screen is SIZE x SIZE pixels
CATEGORIES = ("Travel", "Meals", "Office")
CAT_COLOR = {"Travel": (220, 120, 40), "Meals": (40, 160, 80), "Office": (120, 60, 180)}
AMOUNTS = ("42.50", "18.00", "7.25", "120.00", "63.10")


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass
class Layout:
    """Where each widget sits. ``redesign=True`` moves and renames Submit and changes its internal id."""

    amount: Box
    dropdown: Box
    items: dict[str, Box]
    submit: Box
    help: Box
    popup: Box
    popup_close: Box
    dialog: Box
    dialog_ok: Box
    submit_name: str = "Submit"
    submit_id: str = "btn-submit"
    redesign: bool = False

    @staticmethod
    def sample(rng: random.Random, jitter: bool = True, redesign: bool = False) -> "Layout":
        j = (lambda a: rng.randint(-a, a)) if jitter else (lambda a: 0)
        fx, fy = 8 + j(4), 16 + j(4)
        amount = Box(fx, fy, 40, 10)
        dropdown = Box(fx, fy + 16 + j(2), 40, 10)
        items = {c: Box(dropdown.x, dropdown.y + 11 + 9 * i, 40, 9) for i, c in enumerate(CATEGORIES)}
        if redesign:
            submit = Box(fx, 78 + j(2), 36, 12)
        else:
            submit = Box(58 + j(4), 70 + j(4), 30, 12)
        help_ = Box(80 + j(4), 0, 10, 8)
        px, py = 16 + j(8), 24 + j(8)
        popup = Box(px, py, 56, 40)
        popup_close = Box(px + 40, py + 2, 14, 10)
        dialog = Box(14, 30, 68, 34)
        dialog_ok = Box(36 + j(6), 48 + j(3), 24, 12)
        return Layout(amount, dropdown, items, submit, help_, popup, popup_close, dialog, dialog_ok,
                      submit_name="Send report" if redesign else "Submit",
                      submit_id="action-primary-7" if redesign else "btn-submit", redesign=redesign)


@dataclass
class AppState:
    layout: Layout
    amount: str = ""
    category: str | None = None
    focus: str | None = None
    menu_open: bool = False
    dialog: str | None = None  # None, "error" (submitted too early) or "help"
    popup: bool = False
    toast: bool = False
    records: list[tuple[str, str]] = field(default_factory=list)  # server side, never drawn


class DeskEnv:
    """File one or more expense reports. ``task`` is a list of (category, amount) pairs, done in order.

    ``popup_prob`` is the chance that a promotional pop-up appears once during each report, at a random step.
    ``clear_after_submit=False`` keeps the values on screen after saving, as many real forms do.
    """

    def __init__(self, task, seed=0, jitter=True, redesign=False, popup_prob=0.0, clear_after_submit=True,
                 budget=None):
        self.rng = random.Random(seed)
        self.task = list(task)
        self.popup_prob = popup_prob
        self.clear_after_submit = clear_after_submit
        self.state = AppState(Layout.sample(self.rng, jitter, redesign))
        self.steps = 0
        self.budget = budget if budget is not None else 10 ** 9
        self.log: list[tuple] = []  # the harness's action log
        self._schedule_popup()

    # -- task bookkeeping ------------------------------------------------------------------------------------------
    def _schedule_popup(self):
        self.popup_at = self.steps + self.rng.randint(0, 4) if self.rng.random() < self.popup_prob else None

    @property
    def subtask_index(self) -> int:
        return min(len(self.state.records), len(self.task) - 1)

    @property
    def subtask(self) -> tuple[str, str]:
        return self.task[self.subtask_index]

    def success(self) -> bool:
        """The checker. It reads the saved records: a fact about the world, not about the screen."""
        return self.state.records == self.task

    def done(self) -> bool:
        return len(self.state.records) >= len(self.task) or self.steps >= self.budget

    # -- dynamics ------------------------------------------------------------------------------------------------
    def step(self, action) -> None:
        s, L = self.state, self.state.layout
        self.log.append(action)
        s.toast = False
        if action[0] == "click":
            x, y = action[1], action[2]
            if s.popup:
                if L.popup_close.contains(x, y):
                    s.popup = False
            elif s.dialog:
                if L.dialog_ok.contains(x, y):
                    s.dialog = None
            elif s.menu_open:
                s.menu_open = False
                for c, b in L.items.items():
                    if b.contains(x, y):
                        s.category = c
            elif L.amount.contains(x, y):
                s.focus = "amount"
            elif L.dropdown.contains(x, y):
                s.menu_open, s.focus = True, None
            elif L.submit.contains(x, y):
                s.focus = None
                if s.amount and s.category:
                    n_before = len(s.records)
                    s.records.append((s.category, s.amount))
                    s.toast = True
                    if self.clear_after_submit:
                        s.amount, s.category = "", None
                    if len(s.records) > n_before and len(s.records) < len(self.task):
                        self._schedule_popup()
                else:
                    s.dialog = "error"
            elif L.help.contains(x, y):
                s.dialog, s.focus = "help", None
            else:
                s.focus = None
        elif action[0] == "type":
            if s.focus == "amount" and not (s.popup or s.dialog or s.menu_open):
                s.amount = action[1]
        else:
            raise ValueError(f"unknown action {action!r}")
        self.steps += 1
        if self.popup_at is not None and self.steps >= self.popup_at and not s.popup:
            s.popup, self.popup_at = True, None

    # -- the expert ----------------------------------------------------------------------------------------------
    def expert_action(self):
        """What a careful person would do next from any state, including ones no demonstration shows."""
        s, L = self.state, self.state.layout
        cat, amt = self.subtask
        if s.popup:
            return ("click", *L.popup_close.center)
        if s.dialog:
            return ("click", *L.dialog_ok.center)
        if s.menu_open:
            return ("click", *L.items[cat].center)
        if s.amount != amt:
            if s.focus != "amount":
                return ("click", *L.amount.center)
            return ("type", amt)
        if s.category != cat:
            return ("click", *L.dropdown.center)
        return ("click", *L.submit.center)


EXPERT_STEPS_PER_REPORT = 5  # click amount, type, open menu, pick category, submit


def random_task(rng: random.Random, n_reports: int) -> list[tuple[str, str]]:
    return [(rng.choice(CATEGORIES), rng.choice(AMOUNTS)) for _ in range(n_reports)]


# -- rendering -----------------------------------------------------------------------------------------------------
def _fill(img, b: Box, color):
    img[max(b.y, 0):b.y + b.h, max(b.x, 0):b.x + b.w] = color


def _border(img, b: Box, color):
    img[b.y, b.x:b.x + b.w] = color
    img[b.y + b.h - 1, b.x:b.x + b.w] = color
    img[b.y:b.y + b.h, b.x] = color
    img[b.y:b.y + b.h, b.x + b.w - 1] = color


def _text_bars(img, x, y, n, color):
    """Text as short glyph bars: enough for a small vision model to see 'filled' versus 'empty'."""
    for i in range(n):
        img[y:y + 4, x + 3 * i:x + 3 * i + 2] = color


def render(state: AppState) -> np.ndarray:
    """The screenshot: SIZE x SIZE x 3, uint8. It shows the form but never the saved records."""
    L = state.layout
    img = np.full((SIZE, SIZE, 3), 235, np.uint8)
    img[0:8] = (60, 60, 90)  # title bar
    _fill(img, L.help, (230, 200, 40))
    _fill(img, L.amount, (255, 255, 255))
    _border(img, L.amount, (40, 110, 230) if state.focus == "amount" else (150, 150, 150))
    if state.amount:
        _text_bars(img, L.amount.x + 3, L.amount.y + 3, len(state.amount), (30, 30, 30))
    _fill(img, L.dropdown, (245, 245, 250))
    _border(img, L.dropdown, (150, 150, 150))
    img[L.dropdown.y + 4:L.dropdown.y + 6, L.dropdown.x + L.dropdown.w - 7:L.dropdown.x + L.dropdown.w - 3] = 60
    if state.category:
        _fill(img, Box(L.dropdown.x + 3, L.dropdown.y + 2, 14, 6), CAT_COLOR[state.category])
    _fill(img, L.submit, (40, 90, 200))
    _text_bars(img, L.submit.x + 4, L.submit.y + 4, 5 if not L.redesign else 7, (255, 255, 255))
    if state.menu_open:
        for c, b in L.items.items():
            _fill(img, b, (255, 255, 255))
            _border(img, b, (180, 180, 180))
            _fill(img, Box(b.x + 3, b.y + 2, 10, 5), CAT_COLOR[c])
    if state.toast:
        img[88:94, 10:86] = (60, 170, 90)
    if state.dialog:
        _fill(img, L.dialog, (200, 200, 205) if state.dialog == "error" else (250, 240, 200))
        _border(img, L.dialog, (90, 90, 90))
        _fill(img, L.dialog_ok, (70, 150, 90))
    if state.popup:
        _fill(img, L.popup, (200, 60, 140))
        _text_bars(img, L.popup.x + 4, L.popup.y + 16, 12, (255, 230, 245))
        _fill(img, L.popup_close, (110, 20, 40))
    return img
