#!/usr/bin/env python3
"""Expand src.html into the post. Listings and outputs are pulled from the real files, so they cannot drift."""
import html, json, re, sys
from pathlib import Path

CODE = Path("<CODE>")
SITE = Path("<SITE>")
HERE = Path(__file__).parent
FOLDER = "c03-attention"

def find(lines, spec, after=0):
    """spec may chain anchors with '>>'. Returns 0-based line index."""
    i = after
    for part in spec.split(">>"):
        for j in range(i, len(lines)):
            if lines[j].startswith(part):
                i = j
                break
        else:
            raise SystemExit(f"anchor not found: {part!r} (spec {spec!r})")
    return i

def listing(m):
    a = [x.strip() for x in m.group(1).split("|")]
    src, start, end = a[0], a[1], a[2]
    hl = a[3] if len(a) > 3 else ""
    cap = a[4] if len(a) > 4 else ""
    extra = a[5] if len(a) > 5 else ""
    rel = f"{FOLDER}/{src}"
    lines = (CODE / rel).read_text().split("\n")
    s = find(lines, start)
    e = find(lines, end, s) if end else len(lines) - 2
    code = html.escape("\n".join(lines[s : e + 1]), quote=False)
    h = f' data-highlight="{hl}"' if hl else ""
    c = f"\n  <figcaption>{cap}</figcaption>" if cap else ""
    return f'<figure class="code-listing" data-src="{rel}" data-lines="{s+1}-{e+1}" data-lang="python"{h}{extra}>\n  <pre><code>{code}</code></pre>{c}\n</figure>'

def slice_result(name, start, end):
    text = (CODE / FOLDER / "results" / name).read_text().rstrip("\n")
    lines = text.split("\n")
    s = find(lines, start) if start else 0
    if end:
        e = find(lines, end, s)
    else:
        e = len(lines) - 1
    return "\n".join(lines[s : e + 1])

def out(m):
    a = [x.strip() for x in m.group(1).split("|")]
    name, start, end, frm = a[0], a[1], a[2], a[3]
    label = f' data-label="{a[4]}"' if len(a) > 4 and a[4] else ""
    body = html.escape(slice_result(name, start, end), quote=False)
    return f'<figure class="code-output" data-from="{frm}"{label}>\n  <pre><code>{body}</code></pre>\n</figure>'

def term(m):
    a = [x.strip() for x in m.group(1).split("|")]
    cmd = a[0]
    body = ""
    if len(a) > 1 and a[1]:
        body = "\n" + html.escape(slice_result(a[1], a[2], a[3]), quote=False)
    return f'<figure class="terminal" data-cwd="blog-coding-tutorials">\n  <pre><code>$ {html.escape(cmd, quote=False)}{body}</code></pre>\n</figure>'

src = "".join(p.read_text() for p in sorted(HERE.glob("src_*.html")))
figdata = (CODE / FOLDER / "results" / "figure_data.json").read_text().strip()
src = src.replace("/*@@FIGDATA@@*/null", figdata)
bench = HERE / "chart_data.json"
src = src.replace("/*@@CHARTDATA@@*/null", bench.read_text().strip() if bench.exists() else "null")
src = re.sub(r"^\s*@@LIST (.+)$", listing, src, flags=re.M)
src = re.sub(r"^\s*@@OUT (.+)$", out, src, flags=re.M)
src = re.sub(r"^\s*@@TERM (.+)$", term, src, flags=re.M)
dst = SITE / "blogs" / "build-attention-from-scratch" / "index.html"
dst.parent.mkdir(exist_ok=True)
dst.write_text(src)
print("wrote", dst, len(src))
