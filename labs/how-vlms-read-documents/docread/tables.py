"""A table as rows of cells with spans, written as HTML, OTSL and Markdown, and read back.

Sections "What the model should write down" (OTSL) and "Tables and the TEDS metric" of the post.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser


@dataclass
class Cell:
    text: str
    colspan: int = 1
    rowspan: int = 1


@dataclass
class Table:
    rows: list[list[Cell]] = field(default_factory=list)

    @classmethod
    def from_lists(cls, rows: list[list]) -> "Table":
        """Each cell is a string, or a tuple (text, colspan) or (text, colspan, rowspan)."""
        out = []
        for row in rows:
            out.append([Cell(c) if isinstance(c, str) else Cell(*c) for c in row])
        return cls(out)

    def n_cols(self) -> int:
        return max((sum(c.colspan for c in row) for row in self.rows), default=0)

    def grid(self) -> list[list[tuple[int, int, int]]]:
        """Each grid slot -> (row of the owning cell, its index in that row, 0 if the slot is the cell's top-left)."""
        n_rows, n_cols = len(self.rows), self.n_cols()
        slots: list[list] = [[None] * n_cols for _ in range(n_rows)]
        for r, row in enumerate(self.rows):
            c = 0
            for k, cell in enumerate(row):
                while c < n_cols and slots[r][c] is not None:
                    c += 1
                for dr in range(cell.rowspan):
                    for dc in range(cell.colspan):
                        if r + dr < n_rows and c + dc < n_cols:
                            slots[r + dr][c + dc] = (r, k, 0 if (dr, dc) == (0, 0) else 1 + (dr > 0) + 2 * (dc > 0))
                c += cell.colspan
        return slots


def to_html(table: Table) -> str:
    parts = ["<table>"]
    for row in table.rows:
        parts.append("<tr>")
        for cell in row:
            attrs = (f' colspan="{cell.colspan}"' if cell.colspan > 1 else "") + (
                f' rowspan="{cell.rowspan}"' if cell.rowspan > 1 else "")
            parts.append(f"<td{attrs}>{cell.text}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[Cell]] = []
        self.cell: Cell | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self.cell = Cell("", int(a.get("colspan", 1)), int(a.get("rowspan", 1)))

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.cell.text = self.cell.text.strip()
            self.rows[-1].append(self.cell)
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.text += data


def from_html(html: str) -> Table:
    p = _TableParser()
    p.feed(html)
    return Table(p.rows)


# OTSL: one token per grid slot. C = a cell starts here, L = merged with the left, U = merged with above,
# X = merged both ways, NL = end of row. Cell text follows its C token.
_OTSL_KIND = {0: "C", 2: "U", 3: "L", 4: "X"}


def to_otsl(table: Table) -> list[str]:
    tokens = []
    for r, row in enumerate(table.grid()):
        for slot in row:
            if slot is None:
                tokens.append("C")
                continue
            kind = _OTSL_KIND[slot[2]]
            tokens.append(kind)
            if kind == "C":
                text = table.rows[slot[0]][slot[1]].text
                if text:
                    tokens.append(text)
        tokens.append("NL")
    return tokens


def from_otsl(tokens: list[str]) -> Table:
    grid: list[list[str]] = [[]]
    texts: dict[tuple[int, int], str] = {}
    for tok in tokens:
        if tok == "NL":
            grid.append([])
        elif tok in ("C", "L", "U", "X"):
            grid[-1].append(tok)
        else:
            texts[(len(grid) - 1, len(grid[-1]) - 1)] = tok
    grid = [row for row in grid if row]
    rows = []
    for r, row in enumerate(grid):
        out = []
        for c, kind in enumerate(row):
            if kind != "C":
                continue
            cs = 1
            while c + cs < len(row) and row[c + cs] == "L":
                cs += 1
            rs = 1
            while r + rs < len(grid) and c < len(grid[r + rs]) and grid[r + rs][c] == "U":
                rs += 1
            out.append(Cell(texts.get((r, c), ""), cs, rs))
        rows.append(out)
    return Table(rows)


def to_markdown(table: Table) -> str:
    """Markdown has no merged cells: a spanning cell is written once and its extra slots are left empty."""
    lines = []
    for r, row in enumerate(table.grid()):
        texts = []
        for slot in row:
            texts.append(table.rows[slot[0]][slot[1]].text if slot is not None and slot[2] == 0 else "")
        lines.append("| " + " | ".join(texts) + " |")
        if r == 0:
            lines.append("|" + "---|" * len(row))
    return "\n".join(lines)


def from_markdown(md: str) -> Table:
    rows = []
    for line in md.strip().splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        rows.append([Cell(t.strip()) for t in line.strip("|").split("|")])
    return Table(rows)


def flatten(table: Table) -> str:
    """Every cell's text in row order, separated by spaces: the string a text metric compares."""
    return " ".join(c.text for row in table.rows for c in row if c.text)
