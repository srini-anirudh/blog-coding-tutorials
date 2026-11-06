"""Write one invoice page in six formats, read each back, and measure what survived and what it cost in tokens.

Post section: "What the model should write down", Figure 5 and the formats table. Also the OTSL insight: a table
as an OTSL grid against the same table as HTML.
"""

import collections
import re

import _common
from docread.formats import FORMATS, INVOICE, INVOICE_TABLE, iou, page_words, read_back, write
from docread.metrics import normalized_edit_distance
from docread.tables import to_html, to_markdown, to_otsl
from docread.teds import teds

log = _common.Report("04_output_formats")
try:  # a real subword tokenizer if it is in the local cache; otherwise count whitespace pieces
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    count, tok_name = (lambda s: len(tok(s)["input_ids"])), "SmolLM2 tokenizer (the SmolDocling decoder family)"
except Exception:  # noqa: BLE001
    count, tok_name = (lambda s: len(s.split())), "whitespace pieces (tokenizer not available)"


def count_with_tag_tokens(s: str) -> int:
    """DocTags and OTSL tags (<loc_57>, <fcel>, <nl>, ...) are single tokens in SmolDocling's vocabulary."""
    tags = re.findall(r"<[^>]+>", s)
    return len(tags) + count(re.sub(r"<[^>]+>", " ", s)) if s.strip() else len(tags)


truth_words = page_words()
truth_boxes = [e.box for e in INVOICE]
styled = {("bold", e.text) for e in INVOICE if "bold" in e.style} | {("strike", e.text) for e in INVOICE if "strike" in e.style}
data = {"tokens_counted_with": tok_name, "formats": {}}

log(f"Page: {len(INVOICE)} elements, {len(truth_words)} words, one table with a merged header cell, "
    f"{len(styled)} styled elements (2 bold, 1 struck through). Tokens: {tok_name}.\n")
log(f"{'format':<12} {'tokens':>6} {'word recall':>11} {'order dist':>10} {'headings':>8} {'table TEDS':>10} "
    f"{'box IoU':>8} {'styles':>6}")
for fmt in FORMATS:
    out = write(fmt)
    rec = read_back(fmt, out)
    got = collections.Counter(rec["words"])
    recall = sum(min(got[w], n) for w, n in collections.Counter(truth_words).items()) / len(truth_words)
    order = normalized_edit_distance(rec["words"], truth_words)
    table = teds(rec["table"], INVOICE_TABLE) if rec["table"].rows else 0.0
    box = (sum(max((iou(t, b) for b in rec["boxes"]), default=0.0) for t in truth_boxes) / len(truth_boxes))
    style = len(styled & rec["styled"]) / len(styled)
    n_tok = count_with_tag_tokens(out) if fmt == "DocTags" else count(out)
    row = {"tokens": n_tok, "tokens_plain_tokenizer": count(out), "word_recall": recall, "order_distance": order,
           "headings": len(rec["headings"]), "table_teds": table, "box_iou": box, "style_recall": style, "output": out}
    data["formats"][fmt] = row
    log(f"{fmt:<12} {row['tokens']:>6} {recall:>11.2f} {order:>10.2f} {len(rec['headings']):>5} / 2 {table:>10.2f} "
        f"{box:>8.2f} {len(styled & rec['styled']):>3} / {len(styled)}")
log("DocTags tokens count each tag as one token, as in SmolDocling's vocabulary; the plain SmolLM2 tokenizer")
log(f"splits the tags and gives {data['formats']['DocTags']['tokens_plain_tokenizer']}. "
    "HTML tags are counted with the plain tokenizer.")
log("\nword recall: fraction of the page's words found in the output. order dist: edit distance of the word")
log("sequence to the page's (0 = same order). headings: heading levels recovered. table TEDS: the table parsed back")
log("from the output against the true table. box IoU: mean best overlap of recovered boxes with the true ones.")

log("\nThe table alone, three ways (OTSL structure tokens counted as one token each):")
otsl = " ".join(f"<{t.lower()}>" if t in ("C", "L", "U", "X", "NL") else t for t in to_otsl(INVOICE_TABLE))
for name, s, n in (("HTML", to_html(INVOICE_TABLE), count(to_html(INVOICE_TABLE))),
                   ("OTSL", otsl, count_with_tag_tokens(otsl)),
                   ("Markdown", to_markdown(INVOICE_TABLE).replace("\n", " / "), count(to_markdown(INVOICE_TABLE)))):
    log(f"  {name:<9} {n:>3} tokens  {s if len(s) < 110 else s[:107] + '...'}")
    data.setdefault("table_tokens", {})[name] = n
md_teds = teds(read_back("Markdown", write("Markdown"))["table"], INVOICE_TABLE)
log(f"Read back: HTML and OTSL give TEDS 1.00. Markdown gives {md_teds:.2f}: it has no syntax for the span of 'Amount'.")
log("\nDocTags and layout HTML outputs, for reference:")
for fmt in ("DocTags", "layout HTML"):
    log(f"--- {fmt}")
    for line in write(fmt).splitlines():
        log("  " + line)
log.save(data)
