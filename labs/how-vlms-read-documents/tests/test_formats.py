import pytest

from docread.formats import FORMATS, INVOICE_TABLE, iou, page_words, read_back, write
from docread.teds import teds


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_writes_and_reads_back(fmt):
    rec = read_back(fmt, write(fmt))
    assert rec["words"]


def test_what_survives_each_format():
    rec = {f: read_back(f, write(f)) for f in FORMATS}
    assert set(page_words()) <= set(rec["plain text"]["words"])
    assert len(rec["JSON fields"]["words"]) < len(page_words()) / 2
    assert teds(rec["HTML"]["table"], INVOICE_TABLE) == 1.0
    assert teds(rec["DocTags"]["table"], INVOICE_TABLE) == 1.0
    assert teds(rec["Markdown"]["table"], INVOICE_TABLE) < 1.0
    assert rec["HTML"]["boxes"] == [] and len(rec["layout HTML"]["boxes"]) == 8
    assert min(max(iou(a, b) for b in rec["DocTags"]["boxes"]) for a in rec["layout HTML"]["boxes"]) > 0.9
    assert len(rec["layout HTML"]["styled"]) == 3 and len(rec["Markdown"]["styled"]) == 2
    assert rec["plain text"]["headings"] == [] and len(rec["Markdown"]["headings"]) == 2
