"""One invoice page written in six output formats, then read back to see what survived.

Section "What the model should write down", Figure 5 and the formats table of the post. Each format has a writer
and a reader. A property survives a format only if the reader gets it back from the writer's output.
"""

from __future__ import annotations

import html as htmllib
import json
import re
from dataclasses import dataclass, field

from .tables import Table, from_html, from_markdown, from_otsl, to_html, to_markdown, to_otsl


@dataclass
class Element:
    id: str
    kind: str                       # title, heading, text, table, footer
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 on a 0-1000 page grid
    text: str = ""
    table: Table | None = None
    style: set[str] = field(default_factory=set)  # bold, strike


INVOICE_TABLE = Table.from_lists([
    ["Item", ("Amount", 2)],
    ["", "Net", "Tax"],
    ["Paper", "1,200.00", "216.00"],
    ["Toner", "300.00", "54.00"],
])

INVOICE = [
    Element("e1", "title", (80, 60, 620, 110), "ACME Supplies Ltd.", style={"bold"}),
    Element("e2", "heading", (80, 140, 420, 175), "Invoice INV-2041"),
    Element("e3", "text", (80, 190, 400, 215), "Date: 2026-03-14"),
    Element("e4", "text", (560, 190, 920, 215), "Bill to: Northwind Traders"),
    Element("e5", "table", (80, 250, 920, 470), table=INVOICE_TABLE),
    Element("e6", "text", (80, 500, 600, 525), "Credit note CN-17: -150.00", style={"strike"}),
    Element("e7", "text", (560, 560, 920, 595), "Total due: 1,770.00", style={"bold"}),
    Element("e8", "footer", (440, 950, 560, 970), "Page 1 of 1"),
]
JSON_SCHEMA = {"invoice_number": "INV-2041", "date": "2026-03-14", "bill_to": "Northwind Traders",
               "total_due": "1,770.00"}
FORMATS = ["plain text", "Markdown", "HTML", "JSON fields", "DocTags", "layout HTML"]


def page_words(elements=INVOICE) -> list[str]:
    words = []
    for e in elements:
        if e.table is not None:
            words += [w for row in e.table.rows for c in row for w in c.text.split()]
        else:
            words += e.text.split()
    return words


# ---- writers ---------------------------------------------------------------------------------------------------

def write(fmt: str, elements=INVOICE) -> str:
    out = []
    for e in elements:
        if fmt == "plain text":
            out.append("\n".join(" ".join(c.text for c in row) for row in e.table.rows) if e.table else e.text)
        elif fmt == "Markdown":
            if e.table:
                out.append(to_markdown(e.table))
            else:
                t = f"**{e.text}**" if "bold" in e.style else e.text  # CommonMark has no strikethrough
                out.append({"title": "# ", "heading": "## "}.get(e.kind, "") + t)
        elif fmt == "HTML":
            if e.table:
                out.append(to_html(e.table))
                continue
            t = htmllib.escape(e.text)
            t = f"<b>{t}</b>" if "bold" in e.style else t
            t = f"<s>{t}</s>" if "strike" in e.style else t
            tag = {"title": "h1", "heading": "h2", "footer": "footer"}.get(e.kind, "p")
            out.append(f"<{tag}>{t}</{tag}>")
        elif fmt == "DocTags":
            loc = "".join(f"<loc_{round(v / 2)}>" for v in e.box)  # 0-500 grid, as in SmolDocling
            tag = {"title": "title", "heading": "section_header", "footer": "page_footer"}.get(e.kind, "text")
            if e.table:
                body = "".join(f"<{t.lower()}>" if t in ("C", "L", "U", "X", "NL") else t for t in to_otsl(e.table))
                out.append(f"<otsl>{loc}{body}</otsl>")
            else:
                out.append(f"<{tag}>{loc}{e.text}</{tag}>")
        elif fmt == "layout HTML":
            css = ";".join(s for s in (("font-weight:bold" if "bold" in e.style else ""),
                                        ("text-decoration:line-through" if "strike" in e.style else "")) if s)
            inner = to_html(e.table) if e.table else htmllib.escape(e.text)
            out.append(f'<div data-id="{e.id}" data-kind="{e.kind}" data-bbox="{" ".join(map(str, e.box))}" '
                       f'style="{css}">{inner}</div>')
    if fmt == "JSON fields":
        return json.dumps(JSON_SCHEMA, indent=1)
    return "\n".join(out)


# ---- readers: recover what the output still carries -------------------------------------------------------------

def _strip_tags(s: str) -> str:
    return htmllib.unescape(re.sub(r"<[^>]+>", " ", s))


def read_back(fmt: str, out: str) -> dict:
    """words, heading levels, the table, element boxes and styled texts that can be recovered from `out`."""
    rec = {"words": [], "headings": [], "table": Table(), "boxes": [], "styled": set()}
    if fmt == "plain text":
        rec["words"] = out.split()
        rows = [ln.split() for ln in out.splitlines()[4:8]]  # the reader must guess which lines are the table
        rec["table"] = Table.from_lists(rows)
    elif fmt == "Markdown":
        rec["headings"] = re.findall(r"^(#+) ", out, flags=re.M)
        rec["styled"] = {("bold", t) for t in re.findall(r"\*\*(.+?)\*\*", out)}
        rec["table"] = from_markdown("\n".join(ln for ln in out.splitlines() if ln.startswith("|")))
        rec["words"] = re.sub(r"[#*|]|-{3}", " ", out).split()
    elif fmt == "HTML":
        rec["headings"] = re.findall(r"<(h\d)>", out)
        rec["styled"] = {("bold", _strip_tags(t).strip()) for t in re.findall(r"<b>(.+?)</b>", out)} | {
            ("strike", _strip_tags(t).strip()) for t in re.findall(r"<s>(.+?)</s>", out)}
        rec["table"] = from_html(re.search(r"<table>.*?</table>", out, flags=re.S).group(0))
        rec["words"] = _strip_tags(out).split()
    elif fmt == "JSON fields":
        rec["words"] = " ".join(json.loads(out).values()).split()
    elif fmt == "DocTags":
        rec["headings"] = re.findall(r"<(title|section_header)>", out)
        for m in re.finditer(r"<(\w+)>((?:<loc_\d+>){4})", out):
            rec["boxes"].append(tuple(2 * int(v) for v in re.findall(r"\d+", m.group(2))))
        otsl = re.search(r"<otsl>(?:<loc_\d+>){4}(.*?)</otsl>", out).group(1)
        toks = [t for t in re.split(r"(<\w+>)", otsl) if t]
        rec["table"] = from_otsl([t[1:-1].upper() if t.startswith("<") else t for t in toks])
        rec["words"] = re.sub(r"<[^>]+>", " ", out).split()
    elif fmt == "layout HTML":
        for m in re.finditer(r'data-kind="(\w+)" data-bbox="([\d ]+)" style="([^"]*)">', out):
            rec["boxes"].append(tuple(int(v) for v in m.group(2).split()))
            if m.group(1) in ("title", "heading"):
                rec["headings"].append(m.group(1))
        for m in re.finditer(r'style="([^"]+)">([^<]+)</div>', out):
            if "bold" in m.group(1):
                rec["styled"].add(("bold", htmllib.unescape(m.group(2))))
            if "line-through" in m.group(1):
                rec["styled"].add(("strike", htmllib.unescape(m.group(2))))
        rec["table"] = from_html(re.search(r"<table>.*?</table>", out, flags=re.S).group(0))
        rec["words"] = _strip_tags(out).split()
    return rec


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0
