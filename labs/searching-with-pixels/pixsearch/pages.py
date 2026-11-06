"""Synthetic "pages" made of pixels, with queries whose answer depends on binding.

A page is a 32 x 32 RGB image cut into a 4 x 4 grid of 8 x 8 patches. Six patches hold a glyph: one of
16 shapes drawn in one of 8 colours. The other ten patches are blank (plus noise). A query names two
glyphs as words, for example "red shape3 blue shape7". A page is relevant when it holds both glyphs.

The hard part is binding, the same problem as "2023" and "Asia" on the post's bar chart. Every page has
a *twin*: the same six shapes and the same six colours, but with the colours handed to different shapes
and the glyphs moved to other cells. A model that only knows which shapes and which colours appear on a
page cannot tell a page from its twin. It must know which colour sits on which shape.

Relevance labels are exact: they are computed by set inclusion over the whole corpus, so a query can
have several relevant pages (clones) and nDCG handles that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

N_SHAPES = 16
N_COLORS = 8
N_COMBOS = N_SHAPES * N_COLORS
GRID = 4
PATCH = 8
N_PATCHES = GRID * GRID
ITEMS = 6
IMG = GRID * PATCH

# Query vocabulary: 0 is padding, then colour words, then shape words.
PAD = 0
COLOR_TOK0 = 1
SHAPE_TOK0 = COLOR_TOK0 + N_COLORS
VOCAB = SHAPE_TOK0 + N_SHAPES
QUERY_LEN = 4

COLOR_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "white"]
COLORS = torch.tensor([
    [0.9, 0.1, 0.1], [0.1, 0.8, 0.1], [0.15, 0.25, 0.95], [0.9, 0.9, 0.1],
    [0.85, 0.1, 0.85], [0.1, 0.85, 0.85], [0.95, 0.55, 0.05], [0.95, 0.95, 0.95],
])


def _make_glyphs(seed: int = 1234) -> torch.Tensor:
    """16 distinct random 6 x 6 binary masks, each with 14 to 22 pixels on."""
    g = torch.Generator().manual_seed(seed)
    out: list[torch.Tensor] = []
    while len(out) < N_SHAPES:
        m = (torch.rand(6, 6, generator=g) < 0.5).float()
        if not 14 <= m.sum() <= 22:
            continue
        if all((m != o).sum() >= 8 for o in out):
            out.append(m)
    return torch.stack(out)


GLYPHS = _make_glyphs()


def combo(shape: int, color: int) -> int:
    return shape * N_COLORS + color


@dataclass
class Corpus:
    items: torch.Tensor   # N x ITEMS x 3 (shape, colour, cell), long
    images: torch.Tensor  # N x 3 x 32 x 32, float in [0, 1]

    def __len__(self) -> int:
        return self.items.shape[0]

    @property
    def combos(self) -> torch.Tensor:
        """N x N_COMBOS bool: which (shape, colour) glyphs each page holds."""
        m = torch.zeros(len(self), N_COMBOS, dtype=torch.bool)
        c = self.items[..., 0] * N_COLORS + self.items[..., 1]
        m.scatter_(1, c, True)
        return m


def render(items: torch.Tensor, g: torch.Generator, noise: float = 0.05) -> torch.Tensor:
    """Draw pages. items: N x ITEMS x 3. Each glyph sits in the middle of its patch (a 1-pixel border),
    and Gaussian pixel noise is added everywhere. (With random 0-2 pixel shifts as well, late interaction
    reached only 40 P@1 after 1,500 steps in a trial run, so the lab keeps the glyphs aligned.)"""
    n = items.shape[0]
    img = torch.zeros(n, 3, IMG, IMG)
    for i in range(n):
        for s, c, cell in items[i].tolist():
            r0 = (cell // GRID) * PATCH + 1
            c0 = (cell % GRID) * PATCH + 1
            img[i, :, r0:r0 + 6, c0:c0 + 6] = GLYPHS[s] * COLORS[c][:, None, None]
    img = img + noise * torch.randn(img.shape, generator=g)
    return img.clamp(0, 1)


def _random_items(g: torch.Generator) -> torch.Tensor:
    shapes = torch.randperm(N_SHAPES, generator=g)[:ITEMS]
    while True:  # at least 3 distinct colours, so a twin can always rebind at least half the glyphs
        colors = torch.randint(0, N_COLORS, (ITEMS,), generator=g)
        if len(set(colors.tolist())) >= 3:
            break
    cells = torch.randperm(N_PATCHES, generator=g)[:ITEMS]
    return torch.stack([shapes, colors, cells], dim=1)


def _twin_of(items: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Same shapes, same multiset of colours, colours reassigned so that at least half the
    (shape, colour) bindings change, glyphs moved to new cells."""
    shapes, colors = items[:, 0], items[:, 1]
    while True:
        perm = torch.randperm(ITEMS, generator=g)
        new_colors = colors[perm]
        if (new_colors != colors).sum() >= ITEMS // 2:
            break
    cells = torch.randperm(N_PATCHES, generator=g)[:ITEMS]
    return torch.stack([shapes, new_colors, cells], dim=1)


def _clone_of(items: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Same glyphs, new cells: a second page that answers every query the first one answers."""
    cells = torch.randperm(N_PATCHES, generator=g)[:ITEMS]
    return torch.stack([items[:, 0], items[:, 1], cells], dim=1)


def make_corpus(n_pages: int, g: torch.Generator, kind: str = "twins", clone_frac: float = 0.0,
                noise: float = 0.05) -> Corpus:
    """kind="twins": pages come in (page, twin) pairs, next to each other.
    kind="random": independent pages. clone_frac: that fraction of pages is replaced by a clone of
    the page before it (only used for the false-negative experiment)."""
    rows: list[torch.Tensor] = []
    while len(rows) < n_pages:
        base = _random_items(g)
        rows.append(base)
        if kind == "twins" and len(rows) < n_pages:
            rows.append(_twin_of(base, g))
        if clone_frac and len(rows) < n_pages and torch.rand(1, generator=g).item() < clone_frac:
            rows.append(_clone_of(rows[-1], g))
    items = torch.stack(rows[:n_pages])
    return Corpus(items=items, images=render(items, g, noise))


@dataclass
class Queries:
    tokens: torch.Tensor   # Q x 4, long
    combos: torch.Tensor   # Q x 2, long
    source: torch.Tensor   # Q, the page each query was written from

    def __len__(self) -> int:
        return self.tokens.shape[0]


def tokens_for(c1: int, c2: int) -> list[int]:
    s1, k1 = divmod(c1, N_COLORS)
    s2, k2 = divmod(c2, N_COLORS)
    return [COLOR_TOK0 + k1, SHAPE_TOK0 + s1, COLOR_TOK0 + k2, SHAPE_TOK0 + s2]


def make_queries(corpus: Corpus, g: torch.Generator, pages: torch.Tensor | None = None,
                 binding: bool = True) -> Queries:
    """One query per source page. With binding=True (and kind="twins"), the two glyphs are chosen so
    that the page's twin does not hold both: shapes and colours alone cannot separate the two."""
    if pages is None:
        pages = torch.arange(len(corpus))
    has = corpus.combos
    toks, cmb = [], []
    for p in pages.tolist():
        it = corpus.items[p]
        cs = (it[:, 0] * N_COLORS + it[:, 1]).tolist()
        twin = p ^ 1 if p ^ 1 < len(corpus) else None
        for _ in range(100):
            i, j = torch.randperm(ITEMS, generator=g)[:2].tolist()
            if cs[i] == cs[j]:
                continue
            if binding and twin is not None and has[twin, cs[i]] and has[twin, cs[j]]:
                continue
            break
        toks.append(tokens_for(cs[i], cs[j]))
        cmb.append([cs[i], cs[j]])
    return Queries(torch.tensor(toks), torch.tensor(cmb), pages.clone())


def relevant_sets(corpus: Corpus, queries: Queries) -> list[set[int]]:
    """Exact relevance: every page that holds both glyphs of the query."""
    has = corpus.combos
    both = has[:, queries.combos[:, 0]] & has[:, queries.combos[:, 1]]  # N x Q
    return [set(torch.nonzero(both[:, q]).flatten().tolist()) for q in range(len(queries))]


def describe_query(tokens: torch.Tensor) -> str:
    words = []
    for t in tokens.tolist():
        if COLOR_TOK0 <= t < SHAPE_TOK0:
            words.append(COLOR_NAMES[t - COLOR_TOK0])
        elif t >= SHAPE_TOK0:
            words.append(f"shape{t - SHAPE_TOK0}")
    return " ".join(words)


def training_batch(batch: int, g: torch.Generator, twins: bool = True) -> tuple[Corpus, Queries]:
    """A fresh batch of pages and one binding query per page. With twins=True, each page's twin is in
    the same batch, so in-batch negatives include the hardest possible negative."""
    corpus = make_corpus(batch, g, kind="twins" if twins else "random")
    return corpus, make_queries(corpus, g, binding=twins)
