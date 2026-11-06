#!/usr/bin/env python3
"""Assemble blogs/build-a-modern-llm-from-scratch/index.html from body_*.html, figures.js and real results."""
import html, json, re, sys
from pathlib import Path

CODE = Path("<CODE>")
SITE = Path("<SITE>")
SER = CODE / ".series/code-series/c04-llm"
RES = CODE / "c04-llm/results"
SLUG = "build-a-modern-llm-from-scratch"
TITLE = "Build a Modern LLM From Scratch"
SEARCH = "Build a Modern LLM From Scratch in PyTorch: RMSNorm, RoPE, GQA, SwiGLU, KV Cache"
DESC = "Build a Llama-style language model in PyTorch, step by step, inspect every tensor, then match Hugging Face logits and run real SmolLM2 weights."
SUB = "RMSNorm, RoPE, grouped-query attention, SwiGLU, a KV cache and modern sampling, one tested module at a time. At the end our code reproduces the logits of Hugging Face&rsquo;s Llama and runs real pretrained weights."
TAGS = ["Language models", "PyTorch", "Hugging Face"]

exec((SER / "fills.py").read_text())  # FILLS dict, TREE string

pending = []
def res(name):
    p = RES / name
    if not p.exists():
        pending.append(name)
        return "[pending: %s]" % name
    return p.read_text().rstrip("\n")

def clean(t):
    return "\n".join(l.rstrip() for l in t.splitlines())

def walk(sec):
    t = res("walkthrough.txt")
    m = re.search(r"--- %s ---\n(.*?)(?=\n--- |\Z)" % sec, t, re.S)
    if not m:
        pending.append("W:" + sec); return "[pending]"
    return m.group(1)

def brk(case):
    t = res("break_it.txt")
    out, on = [], False
    for line in t.splitlines():
        if line.startswith("["):
            on = line.startswith("[" + case)
        if on:
            out.append(line)
    if not out:
        pending.append("B:" + case); return "[pending]"
    return "\n".join(out)

def short(name):
    if name == "compare_hf_short":
        return re.sub(r", config ModelConfig\(.*?\)\n", "\n", res("compare_hf.txt"))
    if name == "generate_compare_short":
        return res("generate_compare.txt")
    if name.startswith("bench_generation_"):
        f = {"bench_generation_cpu_short": "bench_generation_cpu.txt", "bench_generation_smol_short": "bench_generation_smollm2_cpu.txt"}[name]
        t = res(f).splitlines()
        return "\n".join(l for l in t if l.startswith("preset") or "total" in l or "speed-up" in l)
    if name == "train_sanity_short":
        return res("train_sanity.txt")
    if name == "pytest_short":
        return res("pytest.txt")
    raise KeyError(name)

def macro(m):
    kind, arg = m.group(1), m.group(2)
    if kind == "W": t = walk(arg)
    elif kind == "B": t = brk(arg)
    elif arg.endswith("_short"): t = short(arg)
    else: t = res(arg)
    return html.escape(clean(t), quote=False)

def fill_code(m):
    head, src, rng = m.group(1), m.group(2), m.group(3)
    a, b = (int(x) for x in rng.split("-"))
    lines = (CODE / src).read_text().splitlines()[a - 1 : b]
    if not lines[0].strip() or not lines[-1].strip():
        print("WARNING blank boundary", src, rng)
    return head + "<pre><code>" + html.escape("\n".join(lines), quote=False) + "</code></pre>"

body = (SER / "body_a.html").read_text() + "\n" + (SER / "body_b.html").read_text()
body = re.sub(r'(<figure class="code-listing" data-src="([^"]+)" data-lines="([^"]+)"[^>]*>\s*)<pre><code>@@</code></pre>', fill_code, body)
assert "@@" not in body
body = re.sub(r"\{\{(W|B|R):([^}]+)\}\}", macro, body)

# Comparison rows from the measured table.
rows = []
heads = res("compare_heads.txt")
for line in heads.splitlines():
    m = re.match(r"(MHA|GQA|MQA)\s+(\d+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+) B\s+([\d.]+)\s+\d+s", line)
    if m:
        rows.append(f'              <tr><th scope="row">{m[1]}</th><td>{m[2]}</td><td>{m[3]}</td><td>{m[5]} bytes</td><td>{m[6]}</td></tr>')
body = body.replace("{{HEADS_ROWS}}", "\n".join(rows) or '              <tr><td colspan="5">[pending]</td></tr>')
ms = re.search(r"steps=(\d+)", heads)
body = body.replace("{{HEADS_STEPS}}", ms[1] if ms else "400")
body = body.replace("{{TREE}}", TREE)
for k, v in FILLS.items():
    body = body.replace("%%" + k + "%%", v)
left = re.findall(r"%%(\w+)%%", body)
if left: pending += left
body = re.sub(r"%%(\w+)%%", "[pending]", body)

# Real data for the figures.
walk_rope = walk("rope")
scores = re.findall(r"score\(q at\s+(\d+), k at\s+(\d+)\), offset (\d+): ([+-][\d.]+)", walk_rope)
rope = []
for m_, n_, off, s in scores:
    extra = {"(5,2)": "Offset 3.", "(103,100)": "Offset 3 again, 98 positions later. Both arrows moved, and the angle between them did not.", "(205,202)": "Offset 3, at position 205. Same angle, same score.", "(5,3)": "Offset 2. The angle changes, so the score changes."}[f"({m_},{n_})"]
    rope.append({"m": int(m_), "n": int(n_), "note": f"<strong>q at {m_}, k at {n_}.</strong> {extra} Measured score over one head: <code>{s}</code>."})
if not rope:
    rope = [{"m": 5, "n": 2, "note": "pending"}]
try:
    samp = json.loads((RES / "sampling_logits.json").read_text())
except FileNotFoundError:
    samp = {"logits": [0.0] * 256}; pending.append("sampling_logits.json")
lat = {"T": [], "off": [], "on": []}
for line in res("bench_generation_mps.txt").splitlines():
    m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
    if m:
        lat["T"].append(int(m[1])); lat["off"].append(float(m[2])); lat["on"].append(float(m[3]))
if not lat["T"]:
    lat = {"T": [0, 1], "off": [0, 1], "on": [0, 1]}
tr = {"step": [0], "train": [None], "val": [None]}
for line in res("train_sanity.txt").splitlines():
    m = re.match(r"step\s+0\s+val ([\d.]+)", line)
    if m: tr["val"][0] = float(m[1])
    m = re.match(r"step\s+(\d+)\s+train ([\d.]+)\s+val ([\d.]+)", line)
    if m:
        tr["step"].append(int(m[1])); tr["train"].append(float(m[2])); tr["val"].append(float(m[3]))
data = {"rope": rope, "sampling": samp, "latency": lat, "training": tr}
figs = (SER / "figures.js").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))

# Template shell.
tpl = (SITE / "docs/code-tutorial-template.html").read_text()
tpl = re.sub(r"\s*<!--.*?-->", "", tpl, flags=re.S)
tpl = re.sub(r'\s*<meta name="robots"[^>]*>', "", tpl)
main_start = tpl.index('<main class="article-content">') + len('<main class="article-content">')
main_end = tpl.index("</main>")
tpl = tpl[:main_start] + "\n" + body + "\n    " + tpl[main_end:]
s_start = tpl.rindex("<script>\n", 0, tpl.index("</body>"))
s_end = tpl.index("</script>", s_start)
tpl = tpl[:s_start] + "<script>\n" + figs + "    " + tpl[s_end:]
css = """/* ---------- article-specific figure styles (keep them light) ---------- */
      #fig-flow svg, #fig-rope svg, #fig-gqa svg, #fig-cache svg, #fig-moe svg { min-width: 620px; }
      #fig-latency svg, #fig-training svg, #sampling-stage svg { min-width: 520px; }
      .params-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 8px 0 4px; color: var(--ink-soft); font: 12.5px/1.5 var(--mono); }
      .legend-item i { display: inline-block; width: 11px; height: 11px; margin-right: 6px; border-radius: 2px; vertical-align: -1px; }
      .legend-item b { color: var(--ink); font-weight: 600; }
      .legend-item em { color: var(--muted); font-style: normal; }
      .params-error { margin: 8px 0; padding: 8px 10px; color: #8d2f1f; background: #fbfaf8; border: 1px solid #dedad2; border-left: 3px solid #b8432f; font: 12.5px/1.5 var(--mono); }
"""
tpl = re.sub(r"/\* -+ article-specific figure styles.*?\.mask-out \{[^}]*\}\n", lambda m: css, tpl, flags=re.S)
text_words = len(re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", " ", body, flags=re.S)).split())
read = max(20, round(text_words / 230))
rep = {
    "[ARTICLE-SLUG]": SLUG, "[DESCRIPTIVE SEARCH AND SOCIAL TITLE]": SEARCH, "[DESCRIPTIVE SEARCH TITLE]": SEARCH,
    "[ONE-SENTENCE ARTICLE DESCRIPTION]": DESC, "[ARTICLE HEADLINE]": TITLE, "[ARTICLE TITLE]": TITLE,
    "[PRIMARY TOPIC]": TAGS[0], "[SECONDARY TOPIC]": TAGS[1], "[THIRD TOPIC]": TAGS[2],
    "[TAG ONE]": TAGS[0], "[TAG TWO]": TAGS[1], "[TAG THREE]": TAGS[2],
    "[ONE- OR TWO-SENTENCE INTRODUCTION: WHAT WE BUILD AND WHAT YOU CAN DO AT THE END]": SUB,
    "[NN min read]": f"{read} min read", "[NN] of 12": "04 of 12", "[PART]": "Language models",
    "[PREVIOUS-SLUG]": "build-attention-from-scratch", "[NEXT-SLUG]": "train-an-llm-from-scratch",
    "Previous &middot; [NN]": "Previous &middot; 03", "Next &middot; [NN]": "Next &middot; 05",
    "[PREVIOUS TITLE]": "Build Attention From Scratch", "[NEXT TITLE]": "Train an LLM From Scratch",
    "[BIBTEX_KEY]": "srinivasan2026modernllm",
}
for k, v in rep.items():
    tpl = tpl.replace(k, v)
leftover = re.findall(r"\[[A-Z][A-Z0-9 -]{3,}\]", tpl)
out = SITE / "blogs" / SLUG / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(tpl)
print(f"wrote {out} ({len(tpl)//1024} KB), words {text_words}, read {read} min")
print("template placeholders left:", sorted(set(leftover)))
print("PENDING:", sorted(set(pending)))
