"""Searching over visual evidence: best-first search over a region tree, with and without a verification step.

The tree follows ZoomEye: the root is the whole photo, each node's children are its four quadrants. A node's score is
the model's own P(Yes) that the region holds a sign with opening hours. Reading a node means asking the question on
that region at the fixed 448-px budget. With verification on, an answer is accepted only if the model, looking at the
same region, says the text is a shop's opening hours. That is the post's relevance check, and it is what rejects the
bus sign.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .answers import FIND, QUESTION, VERIFY, closing_time
from .scene import Scene
from .views import View, render, snap28


@dataclass
class Trace:
    answer: str | None = None
    calls: int = 0  # model calls
    visual_tokens: int = 0  # language-model visual tokens spent, summed over calls
    reads: list = field(default_factory=list)  # (box, answer, accepted)
    rejected: int = 0


def children(view: View, width: int = 448) -> list[View]:
    x0, y0, x1, y1 = view.box
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    boxes = [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)]
    return [View(b, width, snap28(width * (b[3] - b[1]) / (b[2] - b[0]))) for b in boxes]


def _tok(view: View) -> int:
    return (view.enc_w // 14) * (view.enc_h // 14) // 4


def best_first(vlm, scene: Scene, root: View, depth: int = 2, verify: bool = True, max_reads: int = 4,
               min_read_depth: int = 2) -> Trace:
    """Expand the most promising region first. Read leaves (and any node at depth >= min_read_depth)."""
    tr = Trace()
    frontier = [(-1.0, 0, 0, root)]  # (-score, tiebreak, depth, view)
    tie = 1
    while frontier and len(tr.reads) < max_reads:
        _, _, d, v = heapq.heappop(frontier)
        if d >= min_read_depth:
            img = render(scene, v)
            ans = closing_time(vlm.ask([img], QUESTION))
            tr.calls += 1
            tr.visual_tokens += _tok(v)
            ok = ans is not None
            if ok and verify:
                ok = vlm.yes_prob([([img], VERIFY)])[0] > 0.5
                tr.calls += 1
                tr.visual_tokens += _tok(v)
            tr.reads.append((v.box, ans, ok))
            if ok:
                tr.answer = ans
                return tr
            tr.rejected += ans is not None
        if d < depth:
            kids = children(v, root.enc_w)
            scores = vlm.yes_prob([([render(scene, k)], FIND) for k in kids])
            tr.calls += len(kids)
            tr.visual_tokens += sum(_tok(k) for k in kids)
            for k, s in zip(kids, scores):
                heapq.heappush(frontier, (-s, tie, d + 1, k))
                tie += 1
    # Out of budget: fall back on the last answer read, if any.
    tr.answer = next((a for _, a, _ in reversed(tr.reads) if a), None)
    return tr


def exhaustive(vlm, scene: Scene, cells: list[View]) -> Trace:
    """Read every cell, then keep the answer whose region the verifier trusts most."""
    tr = Trace()
    imgs = [render(scene, c) for c in cells]
    answers = [closing_time(a[0]) for a in vlm.generate([([im], QUESTION) for im in imgs])]
    conf = vlm.yes_prob([([im], VERIFY) for im in imgs])
    tr.calls = 2 * len(cells)
    tr.visual_tokens = 2 * sum(_tok(c) for c in cells)
    cands = [(c, a) for c, a in zip(conf, answers) if a is not None]
    tr.answer = max(cands)[1] if cands else None
    return tr


def attention_cell(vlm, scene: Scene, look: View, grid: int = 4) -> tuple[int, int]:
    """The grid cell that gets the most attention from the last prompt position (Zhang and colleagues' idea)."""
    a = vlm.attention_map(render(scene, look), QUESTION)
    gh, gw = a.shape
    best, cell = -1.0, (0, 0)
    for r in range(grid):
        for c in range(grid):
            m = a[r * gh // grid:(r + 1) * gh // grid, c * gw // grid:(c + 1) * gw // grid].sum().item()
            if m > best:
                best, cell = m, (c, r)
    return cell
