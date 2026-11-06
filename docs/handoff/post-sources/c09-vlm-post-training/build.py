#!/usr/bin/env python3
"""Assemble blogs/post-train-a-vlm/index.html from the template, body.html, figures.js and real results."""
import html, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CODE = Path("<CODE>")
FOLDER = CODE / "c09-vlm-post-training"
SITE = Path("<SITE>")
SLUG = "post-train-a-vlm"
TITLE = "Post-Train a VLM"
SEARCH = "Post-Train a VLM: Image-Conditioned DPO, Hallucination Negatives and Visual GRPO"
DESC = "Post-train SmolVLM-256M in code: hallucination negatives on synthetic images, image-conditioned DPO, a blind-preference check, visual GRPO, freeze ablation and reward hacking."
SUB = "We take a real 256M vision-language model and align it by hand: preference pairs with exact truth, image-conditioned DPO, a check that it used the image, visual GRPO, and a reward we hack on purpose."
TAGS = ["Multimodal RL", "Hallucination", "DPO and GRPO"]
READ = sys.argv[1] if len(sys.argv) > 1 else "45 min read"

n_listing = 0
def esc(s): return html.escape(s, quote=False)

def listing(m):
    global n_listing
    parts = [p.strip() for p in m.group(1).split("|")]
    path, start, end = parts[0], parts[1], parts[2]
    hl = parts[3] if len(parts) > 3 else ""
    cap = parts[4] if len(parts) > 4 else ""
    lines = (FOLDER / path).read_text().split("\n")
    a = next((i for i, l in enumerate(lines) if re.search(start, l)), None)
    if a is None: sys.exit(f"no match {start} in {path}")
    if end.isdigit():
        b = a + int(end)
    elif end == "auto":   # top-level block: stop before the next two blank lines or dedent to column 0
        b = a + 1
        while b < len(lines):
            if lines[b].strip() == "" and (b + 1 >= len(lines) or lines[b + 1].strip() == "" or not lines[b + 1].startswith((" ", "\t"))):
                break
            b += 1
    else:
        b = next(i for i in range(a, len(lines)) if re.search(end, lines[i])) + 1
    while lines[b - 1].strip() == "": b -= 1
    n_listing += 1
    code = esc("\n".join(lines[a:b]))
    hattr = f' data-highlight="{hl}"' if hl else ""
    src = f"c09-vlm-post-training/{path}"
    capt = f'<figcaption><span class="ct-fig-num">Listing {n_listing}</span>{cap}</figcaption>' if cap else ""
    return f'<figure class="code-listing" data-src="{src}" data-lines="{a+1}-{b}" data-lang="python"{hattr}>\n<pre><code>{code}</code></pre>\n{capt}</figure>'

NOISE = re.compile(r"Redirects are currently|Loading weights|unauthenticated requests|^W\d{4}|HF_TOKEN")
def output(m):
    parts = [p.strip() for p in m.group(1).split("|")]
    path, label = parts[0], parts[1]
    if not (FOLDER / path).exists():
        return f'<figure class="code-output" data-from="{esc(label)}"><pre><code>Run pending.</code></pre></figure>'
    lines = [l for l in (FOLDER / path).read_text().split("\n") if not NOISE.search(l)]
    if len(parts) > 2 and parts[2]:
        a = next((i for i, l in enumerate(lines) if re.search(parts[2], l)), 0)
        lines = lines[a:]
    if len(parts) > 3 and parts[3]:
        b = next((i for i, l in enumerate(lines) if re.search(parts[3], l)), len(lines) - 1)
        lines = lines[:b + 1]
    if len(parts) > 4 and parts[4]:
        rx = parts[4].replace("~", "|")
        lines = [l for l in lines if re.search(rx, l)]
    while lines and not lines[-1].strip(): lines.pop()
    while lines and not lines[0].strip(): lines.pop(0)
    return f'<figure class="code-output" data-from="{esc(label)}">\n<pre><code>{esc(chr(10).join(lines))}</code></pre>\n</figure>'

def load(name):
    p = FOLDER / "results" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None

def thin(log, keys):
    return [{k: (round(e[k], 4) if isinstance(e[k], float) else e[k]) for k in keys if k in e} for e in log]

def data():
    d = {"probe": load("blind_probe"), "rollouts": load("show_rollouts"), "perception": load("perception_probe")}
    for key, name in (("dpoBal", "dpo_balanced"), ("dpoBias", "dpo_biased")):
        r = load(name)
        if r: d[key] = {"before": r["before"], "after": r["after"], "log": thin(r["log"], ["step", "loss", "margin"]),
                        "tb": r.get("tokens_before"), "ta": r.get("tokens_after"), "seconds": r["train_seconds"]}
    d["grpo"] = {}
    for key, name in (("llm", "grpo_count_exact_llm"), ("projector", "grpo_count_exact_projector_llm"), ("all", "grpo_count_exact_all")):
        r = load(name)
        if r: d["grpo"][key] = {"trainable": r["trainable"], "before": r.get("before"), "after": r.get("after"), "delta": r.get("delta_norm"),
                                "seconds": r["train_seconds"], "log": thin(r["log"], ["step", "reward", "zero_std_groups"])}
    d["hack"] = {}
    for key, name in (("recall", "grpo_caption_color_recall_llm"), ("f1", "grpo_caption_color_f1_llm")):
        r = load(name)
        if r: d["hack"][key] = {"before": r.get("before"), "after": r.get("after"), "seconds": r["train_seconds"],
                                "log": [dict(thin([e], ["step", "reward", "strict", "n_colors", "hallucinated", "length"])[0], s=e["sample"][0]) for e in r["log"]]}
    return d

tpl = (SITE / "docs/code-tutorial-template.html").read_text()
body = (HERE / "body.html").read_text()
fig = (HERE / "figures.js").read_text()
tree = (HERE / "tree.txt").read_text().rstrip()
body = body.replace("[[TREE]]", tree)
prose = {}
pf = HERE / "prose.html"
if pf.exists():
    for chunk in re.split(r"^@@", pf.read_text(), flags=re.M)[1:]:
        key, _, val = chunk.partition("\n")
        prose[key.strip()] = val.strip()
body = re.sub(r"\[\[([A-Z_]+)\]\]", lambda m: prose.get(m.group(1), "<p><em>Result pending.</em></p>"), body)
body = re.sub(r"\[\[L (.+?)\]\]", listing, body, flags=re.S)
body = re.sub(r"\[\[O (.+?)\]\]", output, body, flags=re.S)

out = tpl
out = re.sub(r"\s*<!--\s*=+.*?DRIFT CHECKER.*?-->", "", out, count=1, flags=re.S)
out = out.replace('    <!-- This template is intentionally excluded from search. REMOVE this line in a published article. -->\n    <meta name="robots" content="noindex, nofollow" />\n', "")
out = re.sub(r"<main class=\"article-content\">.*</main>", lambda m: '<main class="article-content">\n' + body + "\n    </main>", out, flags=re.S)
out = re.sub(r"<!-- Article figures.*?</script>\n  </body>", lambda m: "<script>\nwindow.C9_DATA = " + json.dumps(data(), separators=(",", ":")) + ";\n" + fig + "\n    </script>\n  </body>", out, flags=re.S)
out = re.sub(r"/\* -+ article-specific figure styles.*?\.mask-out \{[^\n]*\n", lambda m: (HERE / "styles.css").read_text() + "\n", out, flags=re.S)
for a, b in {"[ONE-SENTENCE ARTICLE DESCRIPTION]": DESC, "[DESCRIPTIVE SEARCH AND SOCIAL TITLE]": SEARCH, "[DESCRIPTIVE SEARCH TITLE]": SEARCH,
             "[ARTICLE-SLUG]": SLUG, "[ARTICLE HEADLINE]": TITLE + ": image-conditioned DPO, hallucination negatives and visual GRPO", "[PRIMARY TOPIC]": "vision-language models",
             "[SECONDARY TOPIC]": "image-conditioned DPO", "[THIRD TOPIC]": "visual GRPO", "[ARTICLE TITLE]": TITLE,
             "[ONE- OR TWO-SENTENCE INTRODUCTION: WHAT WE BUILD AND WHAT YOU CAN DO AT THE END]": SUB,
             "[TAG ONE]": TAGS[0], "[TAG TWO]": TAGS[1], "[TAG THREE]": TAGS[2], "[NN min read]": READ, "[BIBTEX_KEY]": "srinivasan2026posttrainvlm",
             "[NN] of 12 &middot; [PART]": "09 of 12 &middot; Vision-language models", "[PREVIOUS-SLUG]": "train-a-real-vlm", "Previous &middot; [NN]": "Previous &middot; 08",
             "[PREVIOUS TITLE]": "Train a Real VLM", "[NEXT-SLUG]": "build-an-agent-harness-from-scratch", "Next &middot; [NN]": "Next &middot; 10",
             "[NEXT TITLE]": "Build an Agent Harness From Scratch"}.items():
    out = out.replace(a, b)
left = re.findall(r"\[[A-Z][A-Z -]{3,}\]", out)
if left: print("PLACEHOLDERS LEFT:", left)
dest = SITE / "blogs" / SLUG / "index.html"
dest.write_text(out)
print("wrote", dest, len(out), "listings", n_listing)
