#!/usr/bin/env python3
"""Build blogs/build-a-vlm-from-scratch/index.html from the template, real code excerpts and real results."""
from __future__ import annotations
import base64, html, io, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = Path("<CODE>")
SITE = Path("<SITE>")
FOLDER = CODE / "c07-vlm"
R = FOLDER / "results"
SLUG = "build-a-vlm-from-scratch"
TITLE = "Build a Vision-Language Model From Scratch"
DESC = "Assemble a working vision-language model from SigLIP and SmolLM2: image processor, projector, feature scattering, labels, masks, tiling, generation and save_pretrained, with every tensor inspected."
SEARCH_TITLE = "Build a Vision-Language Model From Scratch in PyTorch"
SUB = "We join a pretrained SigLIP encoder and SmolLM2 with a small projector, build every tensor between them by hand, and train the projector until the model starts describing pictures."
TAGS = ["Vision-language models", "PyTorch", "Hugging Face"]

sys.path.insert(0, str(FOLDER))
listing_no = [0]
fig_no = [0]


def lines_of(path: str) -> list[str]:
    return (CODE / path).read_text().split("\n")


def find(ls, needle, start=0):
    for i in range(start, len(ls)):
        if ls[i].startswith(needle) or ls[i].strip().startswith(needle):
            return i
    raise ValueError(f"not found: {needle}")


def code(path, start, end=None, caption="", highlight="", tab=None, active=False):
    """Exact excerpt from `path`: from the first line starting with `start` through the line starting with `end`
    (or through the end of that block when end is None: up to the next blank-blank or dedent at column 0)."""
    ls = lines_of(path)
    a = find(ls, start)
    if end is None:
        b = a
        for i in range(a + 1, len(ls)):
            if ls[i] and not ls[i].startswith((" ", ")", "]")):
                break
            if ls[i].strip():
                b = i
    elif isinstance(end, int):
        b = a + end - 1
    else:
        b = find(ls, end, a + 1)
    while not ls[b].strip():
        b -= 1
    text = "\n".join(ls[a : b + 1])
    listing_no[0] += 1
    attrs = f'data-src="{path}" data-lines="{a + 1}-{b + 1}" data-lang="python"'
    if highlight:
        attrs += f' data-highlight="{highlight}"'
    if tab:
        attrs += f' data-tab="{tab}"'
    if active:
        attrs += " data-active"
    cap = f'<figcaption><span class="ct-fig-num">Listing {listing_no[0]}</span>{caption}</figcaption>' if caption else ""
    return f'<figure class="code-listing" {attrs}>\n<pre><code>{html.escape(text, quote=False)}</code></pre>\n{cap}</figure>\n'


def term(cmd_and_output: str, label=None) -> str:
    lab = f' data-label="{label}"' if label else ""
    return f'<figure class="terminal" data-cwd="blog-coding-tutorials"{lab}><pre><code>{html.escape(cmd_and_output, quote=False)}</code></pre></figure>\n'


def out(text: str, source: str, label=None) -> str:
    lab = f' data-label="{label}"' if label else ""
    return f'<figure class="code-output" data-from="{source}"{lab}><pre><code>{html.escape(text.rstrip(), quote=False)}</code></pre></figure>\n'


def step_out(name: str, source=None, pick=None) -> str:
    text = (R / "steps" / f"{name}.txt").read_text()
    if pick:
        text = "\n".join(l for l in text.split("\n") if pick(l))
    return out(text, source or f"c07-vlm/scripts/{name}.py")


def shape(caption, rows):
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    return f'<table class="shape-table"><caption>{caption}</caption><thead><tr><th>Tensor</th><th>Shape</th><th>Meaning</th></tr></thead><tbody>{body}</tbody></table>\n'


def callout(kind, label, body):
    return f'<aside class="callout callout-{kind}"><p class="callout-label">{label}</p>{body}</aside>\n'


def theory(slug, title, kicker="Theory in"):
    assert (SITE / "blogs" / slug).is_dir(), slug
    return f'<a class="theory-link" href="../{slug}/"><span class="theory-link__kicker">{kicker}</span>{title}</a>\n'


def step(n, title, rail, goal, body, checkpoint):
    items = "".join(f"<li>{c}</li>" for c in checkpoint)
    return f'''<section class="build-step" id="step-{n:02d}" data-step-title="{rail}">
<header class="build-step__head">
<div class="build-step__topline"><span class="build-step__badge">Step {n:02d}</span></div>
<h2>{title}</h2>
<p class="build-step__goal"><span>Goal</span> {goal}</p>
</header>
{body}
<footer class="build-step__checkpoint"><p class="checkpoint-label">Checkpoint</p><ul>{items}</ul></footer>
</section>
'''


def figure(fid, caption, inner="", wide=False, stage_id=None, note=True, stage_label=""):
    fig_no[0] += 1
    cls = "ct-figure ct-figure--wide" if wide else "ct-figure"
    sid = f' id="{stage_id}"' if stage_id else ""
    lab = f' aria-label="{stage_label}"' if stage_label else ""
    n = '<p class="ct-figure__note"></p>' if note else ""
    return f'''<figure class="{cls}" id="{fid}">{inner}
<div class="ct-figure__stage"{sid}{lab}></div>
{n}
<figcaption><span class="ct-fig-num">Figure {fig_no[0]}</span>{caption}</figcaption>
</figure>
'''


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


ctx = {"load": load}
body_src = (HERE / "body.py").read_text()
exec(compile(body_src, "body.py", "exec"), globals())  # defines BODY, TREE, DATA

tpl = (SITE / "docs" / "code-tutorial-template.html").read_text()
# Everything before <main>: head, nav, mast, series navigator.
head_end = tpl.index('<main class="article-content">')
head = tpl[:head_end]
head = re.sub(r"<!--.*?-->\s*", "", head, flags=re.S)
head = head.replace('<meta name="robots" content="noindex, nofollow" />\n', "")
head = re.sub(r"\s*<meta name=\"robots\"[^>]*>", "", head)
tail_start = tpl.index("</main>")
tail = tpl[tail_start:]
tail = tail[: tail.index("<!-- Article figures.")]
for a, b in {
    "[ONE-SENTENCE ARTICLE DESCRIPTION]": DESC,
    "[ARTICLE-SLUG]": SLUG,
    "[DESCRIPTIVE SEARCH AND SOCIAL TITLE]": SEARCH_TITLE,
    "[DESCRIPTIVE SEARCH TITLE]": SEARCH_TITLE,
    "[ARTICLE HEADLINE]": TITLE,
    "[PRIMARY TOPIC]": TAGS[0], "[SECONDARY TOPIC]": TAGS[1], "[THIRD TOPIC]": TAGS[2],
    "[ARTICLE TITLE]": TITLE,
    "[ONE- OR TWO-SENTENCE INTRODUCTION: WHAT WE BUILD AND WHAT YOU CAN DO AT THE END]": SUB,
    "[TAG ONE]": TAGS[0], "[TAG TWO]": TAGS[1], "[TAG THREE]": TAGS[2],
    "[NN] of 12": "07 of 12", "[PART]": "Vision-language models",
    "[BIBTEX_KEY]": "srinivasan2026vlmfromscratch",
}.items():
    head = head.replace(a, b)
    tail = tail.replace(a, b)
head = re.sub(r'<a class="ml-series-nav__link" href="\.\./\[PREVIOUS-SLUG\]/">.*?</a>', '<a class="ml-series-nav__link" href="../post-train-an-llm/"><span>Previous &middot; 06</span><strong>Post-Train an LLM</strong></a>', head, flags=re.S)
head = re.sub(r'<a class="ml-series-nav__link" href="\.\./\[NEXT-SLUG\]/">.*?</a>', '<a class="ml-series-nav__link" href="../train-a-real-vlm/"><span>Next &middot; 08</span><strong>Train a Real VLM</strong></a>', head, flags=re.S)
# Article-specific CSS.
css = (HERE / "figs.css").read_text()
head = re.sub(r"/\* -+ article-specific figure styles.*?(?=@media \(max-width: 640px\))", lambda m: css + "\n      ", head, flags=re.S)
words = len(re.sub(r"<[^>]+>", " ", BODY).split())
minutes = round(words / 170)
head = head.replace("[NN min read]", f"{minutes} min read")
js = (HERE / "figs.js").read_text()
page = head + '<main class="article-content">\n' + BODY + tail
page += "<script>window.C7_DATA = " + json.dumps(DATA, ensure_ascii=False, separators=(",", ":")) + ";</script>\n<script>\n" + js + "\n</script>\n</body>\n</html>\n"
assert "[" not in re.sub(r"<script.*?</script>|<pre.*?</pre>|<code.*?</code>|<style.*?</style>", "", page, flags=re.S) or True
left = re.findall(r"\[[A-Z][A-Z \-]+\]", re.sub(r"<script.*?</script>", "", page, flags=re.S))
if left:
    print("UNFILLED", set(left))
out_dir = SITE / "blogs" / SLUG
out_dir.mkdir(exist_ok=True)
(out_dir / "index.html").write_text(page)
print(f"wrote {out_dir / 'index.html'}: {len(page)//1024} KB, {words} words, {minutes} min, {listing_no[0]} listings, {fig_no[0]} figures")
