"""Five ways to present one MiniDesk screen to a model, and what each one costs.

Token counts for text views use a fixed rule of thumb (4 characters per token), not a real tokenizer. The screenshot
cost is the number of visual tokens after fitting the image to a patch grid.
"""

from __future__ import annotations

import math

import numpy as np

from .desk import CATEGORIES, AppState, Box, render


def elements(state: AppState) -> list[dict]:
    """The interactive elements that are actually on screen, in document order."""
    L = state.layout
    out = []
    if L.redesign:  # the redesign also moves Submit to the top of the document
        out.append(dict(id=L.submit_id, role="button", name=L.submit_name, box=L.submit))
    out += [dict(id="help", role="link", name="Help", box=L.help),
            dict(id="amount", role="textbox", name="Amount", box=L.amount, value=state.amount,
                 focused=state.focus == "amount"),
            dict(id="category", role="combobox", name="Category", box=L.dropdown, value=state.category or "")]
    if state.menu_open:
        out += [dict(id=f"opt-{c.lower()}", role="option", name=c, box=L.items[c]) for c in CATEGORIES]
    if not L.redesign:
        out.append(dict(id=L.submit_id, role="button", name=L.submit_name, box=L.submit))
    if state.dialog:
        out.append(dict(id="dialog-ok", role="button", name="OK", box=L.dialog_ok))
    if state.popup:
        out.append(dict(id="popup-close", role="button", name="Close", box=L.popup_close))
    return out


def a11y_tree(state: AppState) -> str:
    lines = ["window 'Expenses'"]
    for e in elements(state):
        b = e["box"]
        extra = f" value={e['value']!r}" if "value" in e else ""
        extra += " focused" if e.get("focused") else ""
        lines.append(f"  [{e['role']}] {e['name']!r}{extra} @({b.x},{b.y},{b.w},{b.h})")
    return "\n".join(lines)


def dom(state: AppState) -> str:
    """A web-page view: exact, but padded with styles, scripts and hidden elements, as real pages are."""
    head = ("<html><head><style>.btn{padding:4px 8px;border-radius:3px}.primary{background:#285ac8;color:#fff}"
            ".field{border:1px solid #969696}.hidden{display:none}</style>"
            "<script src='/static/analytics.js'></script><script>window.__STATE__={csrf:'x9f2',v:3}</script>"
            "</head><body>")
    body = ["<div class='hidden' id='tracking-pixel'></div>", "<input type='hidden' name='csrf' value='x9f2'>"]
    for e in elements(state):
        b = e["box"]
        style = f"style='position:absolute;left:{b.x}px;top:{b.y}px;width:{b.w}px;height:{b.h}px'"
        if e["role"] == "textbox":
            body.append(f"<input id='{e['id']}' class='field' aria-label='{e['name']}' value='{e['value']}' {style}>")
        elif e["role"] == "combobox":
            opts = "".join(f"<option>{c}</option>" for c in CATEGORIES)
            body.append(f"<select id='{e['id']}' class='field' {style}>{opts}</select>")
        else:
            body.append(f"<button id='{e['id']}' class='btn primary' {style}>{e['name']}</button>")
    return head + "".join(body) + "</body></html>"


def set_of_mark(state: AppState) -> tuple[np.ndarray, dict[int, Box]]:
    """The screenshot with a numbered mark on each element. Here the marks come from the element list."""
    img = render(state).copy()
    marks = {}
    for i, e in enumerate(elements(state), start=1):
        b = e["box"]
        img[b.y:b.y + 3, b.x:b.x + 3] = (255, 0, 0)  # a red corner tag stands in for the drawn number
        marks[i] = b
    return img, marks


def visual_tokens(w: int, h: int, patch: int = 14, max_tokens: int = 1024) -> int:
    """Visual tokens for a w x h screenshot after it is shrunk (never enlarged) to fit max_tokens patches."""
    n = math.ceil(w / patch) * math.ceil(h / patch)
    if n <= max_tokens:
        return n
    s = math.sqrt(max_tokens / n)
    return math.floor(w * s / patch) * math.floor(h * s / patch)


def text_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def observation_costs(state: AppState, screen_w: int = 1920, screen_h: int = 1080) -> dict[str, int]:
    """Approximate prompt tokens per observation, for a screen of the given real size."""
    shot = visual_tokens(screen_w, screen_h)
    legend = "\n".join(f"{i}: {e['role']} {e['name']}" for i, e in enumerate(elements(state), 1))
    compact = "\n".join(f"{e['role']} {e['name']!r} @{e['box'].center}" for e in elements(state))
    return {"screenshot": shot, "a11y_tree": text_tokens(a11y_tree(state)), "dom": text_tokens(dom(state)),
            "set_of_mark": shot + text_tokens(legend), "hybrid": shot + text_tokens(compact)}


# -- locators written against the original page, then run on the redesigned one -----------------------------------
def locate_dom_by_id(state, element_id="btn-submit"):
    return next((e["box"].center for e in elements(state) if e["id"] == element_id), None)


def locate_a11y_by_name(state, role="button", name="Submit"):
    return next((e["box"].center for e in elements(state) if e["role"] == role and e["name"] == name), None)


def locate_mark(state, mark: int):
    _, marks = set_of_mark(state)
    return marks[mark].center if mark in marks else None


def locate_fixed_point(state, point):
    return point


def locate_primary_colour(state):
    """A pixel locator: the centre of the blue primary-button colour. A hand-written stand-in for a learned grounder."""
    img = render(state).astype(int)
    m = (abs(img[..., 0] - 40) < 8) & (abs(img[..., 1] - 90) < 8) & (abs(img[..., 2] - 200) < 8)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())
