#!/usr/bin/env python3
"""Integrate ready Coding Tutorials posts into the site (idempotent). Usage: cpublish.py slug [slug ...] | --all"""
import html, os, pathlib, re, sys
SITE = pathlib.Path(os.environ["SITE"])
POSTS = [
 ("pytorch-from-first-principles","PyTorch From First Principles","Raw mechanics","c01-pytorch"),
 ("the-hugging-face-stack","The Hugging Face Stack","Raw mechanics","c02-hf-stack"),
 ("build-attention-from-scratch","Build Attention From Scratch","Raw mechanics","c03-attention"),
 ("build-a-modern-llm-from-scratch","Build a Modern LLM From Scratch","Language models","c04-llm"),
 ("train-an-llm-from-scratch","Train an LLM From Scratch","Language models","c05-pretraining"),
 ("post-train-an-llm","Post-Train an LLM","Language models","c06-post-training"),
 ("build-a-vlm-from-scratch","Build a Vision-Language Model From Scratch","Vision-language models","c07-vlm"),
 ("train-a-real-vlm","Train a Real VLM","Vision-language models","c08-vlm-training"),
 ("post-train-a-vlm","Post-Train a VLM","Vision-language models","c09-vlm-post-training"),
 ("build-an-agent-harness-from-scratch","Build an Agent Harness From Scratch","Agents","c10-agent-harness"),
 ("add-memory-to-the-agent","Add Memory and Context Engineering to the Agent","Agents","c11-agent-memory"),
 ("train-an-agent-with-rl","Train an Agent With Reinforcement Learning","Agents","c12-agent-rl"),
]
args = sys.argv[1:]
ready = {p[0] for p in POSTS} if args == ["--all"] else set(args)
R = [p for p in POSTS if p[0] in ready]
for p in R: assert (SITE/"blogs"/p[0]/"index.html").exists(), p[0]
num = lambda p: f"{POSTS.index(p)+1:02d}"
BASE = "https://srini-anirudh.github.io/blogs/"

def nav(p):
    i = R.index(p); l = ""
    if i > 0: q = R[i-1]; l += f'<a class="ml-series-nav__link" href="../{q[0]}/"><span>Previous &middot; {num(q)}</span><strong>{q[1]}</strong></a>'
    if i < len(R)-1: q = R[i+1]; l += f'<a class="ml-series-nav__link" href="../{q[0]}/"><span>Next &middot; {num(q)}</span><strong>{q[1]}</strong></a>'
    return ('<nav class="ml-series-nav" aria-label="Coding Tutorials series">\n'
            '      <a class="ml-series-nav__identity" href="../index.html?series=code">Coding Tutorials</a>\n'
            f'      <span class="ml-series-nav__position">{num(p)} of {len(POSTS):02d} &middot; {p[2]}</span>\n'
            f'      <div class="ml-series-nav__links">{l}</div>\n    </nav>')
NAV = re.compile(r'<nav class="ml-series-nav" aria-label="Coding Tutorials series">.*?</nav>', re.S)
strip = lambda x: html.unescape(re.sub(r"<[^>]+>", "", x)).strip()
def meta(p):
    s = (SITE/"blogs"/p[0]/"index.html").read_text()
    tags = re.search(r'class="article-taxonomy[^"]*"[^>]*>(.*?)</', s, re.S)
    tx = re.search(r'<[a-z]+[^>]*class="article-taxonomy[^"]*"[^>]*>(.*?)</(?:div|ul|p|nav)>', s, re.S)
    tl = [strip(t) for t in re.findall(r'<(?:span|li|a)[^>]*>(.*?)</(?:span|li|a)>', tx.group(1), re.S)][:3] if tx else []
    return dict(title=strip(re.search(r"<h1[^>]*>(.*?)</h1>", s, re.S).group(1)),
                summary=html.unescape(re.search(r'<meta name="description" content="([^"]*)"', s).group(1)),
                read=(re.search(r"(\d+)\s*min read", s) or [None,"30"])[1], tags=tl)
for p in R:
    f = SITE/"blogs"/p[0]/"index.html"; s = f.read_text()
    s = NAV.sub(lambda m: nav(p), s, count=1) if NAV.search(s) else s.replace("</header>", "</header>\n    "+nav(p), 1)
    f.write_text(s)
def card(p):
    d = meta(p); sl = p[0]
    gif = f' data-animated-src="{sl}/preview.gif"' if (SITE/"blogs"/sl/"preview.gif").exists() else ""
    tags = "".join(f"<span>{html.escape(x)}</span>" for x in d["tags"])
    return f'''          <a class="blog-card" href="{sl}/" target="_self" data-blog-categories="code" data-blog-series="code">
            <picture class="blog-card-media">
              <img src="{sl}/preview.png" data-still-src="{sl}/preview.png"{gif} width="960" height="540" loading="lazy" decoding="async" alt="{html.escape(d['title'])}: preview of the tutorial's main figure" />
            </picture>
            <div class="blog-card-body">
              <span class="blog-card-label">Coding tutorial &middot; C{POSTS.index(p)+1} &middot; {p[2]}</span>
              <h3>{html.escape(d['title'])}</h3>
              <p>{html.escape(d['summary'])}</p>
              <div class="blog-meta"><span>Published <time datetime="2026-11-06">November 6, 2026</time></span><span>{d['read']} min read</span></div>
              <div class="blog-tags">{tags}</div>
            </div>
          </a>
'''
idx = SITE/"blogs"/"index.html"; s = idx.read_text()
block = "<!-- code-series:start -->\n" + "".join(card(p) for p in R) + "          <!-- code-series:end -->"
if "<!-- code-series:start -->" in s:
    s = re.sub(r"<!-- code-series:start -->.*?<!-- code-series:end -->", lambda m: block, s, flags=re.S)
else:
    s = s.replace("<!-- vlm-series:start -->", block + "\n          <!-- vlm-series:start -->", 1)
groups = {}
for p in R: groups.setdefault(p[2], []).append(p[0])
paths = ";".join(f"{k}::{' '.join(v)}" for k, v in groups.items())
btn = ('<button type="button" class="blog-filter" data-blog-filter="code" data-blog-filter-label="Code" '
       f'data-blog-members="code" data-blog-paths="{paths}" '
       'data-path-description="Coding tutorials: build it step by step, from a tensor to a foundation model and an agent. The code is in the blog-coding-tutorials repository." aria-pressed="false">Code</button>')
if 'data-blog-filter="code"' in s:
    s = re.sub(r'<button type="button" class="blog-filter" data-blog-filter="code".*?</button>', lambda m: btn, s, flags=re.S)
else:
    m = re.search(r'(\s*)<button type="button" class="blog-filter" data-blog-filter="research"', s)
    s = s[:m.start()] + m.group(1) + btn + s[m.start():]
entry = f'{{ "@type": "CreativeWorkSeries", "@id": "{BASE}#code", "name": "Coding Tutorials", "numberOfItems": {len(R)} }}'
if f'{BASE}#code"' in s:
    s = re.sub(r'\{ "@type": "CreativeWorkSeries", "@id": "' + re.escape(BASE) + r'#code".*?\}', lambda m: entry, s)
else:
    s = s.replace('"name": "Grounded Document Intelligence", "numberOfItems": 5 }', '"name": "Grounded Document Intelligence", "numberOfItems": 5 },\n          ' + entry, 1)
n = len(re.findall(r'<a class="blog-card[ "]', s))
s = re.sub(r"Showing all \d+ articles", f"Showing all {n} articles", s)
idx.write_text(s)
f = SITE/"sitemap.xml"; s = f.read_text()
for p in R:
    loc = BASE+p[0]+"/"
    if loc not in s: s = s.replace("</urlset>", f"  <url><loc>{loc}</loc><lastmod>2026-11-06</lastmod></url>\n</urlset>", 1)
f.write_text(s)
f = SITE/"feed.xml"; s = f.read_text()
for p in reversed(R):
    loc = BASE+p[0]+"/"
    if loc in s: continue
    d = meta(p)
    s = s.replace("    <item>\n", f"    <item>\n      <title>{html.escape(d['title'])}</title>\n      <link>{loc}</link>\n      <guid isPermaLink=\"true\">{loc}</guid>\n      <pubDate>Fri, 06 Nov 2026 00:00:00 +0530</pubDate>\n      <description>{html.escape(d['summary'])}</description>\n    </item>\n    <item>\n", 1)
f.write_text(s)
f = SITE/".github/workflows/indexnow.yml"; s = f.read_text()
for p in R:
    path = f'"/blogs/{p[0]}/"'
    if path not in s: s = s.replace('"/blogs/building-grounded-document-intelligence/",', f'"/blogs/building-grounded-document-intelligence/", {path},', 1)
f.write_text(s)
print("ready:", [num(p) for p in R], "cards:", n)
