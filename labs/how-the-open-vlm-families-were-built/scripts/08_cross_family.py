"""Section "What correlates with capability": the cross-family claims, recomputed from the post's own Table 1.

Offline. data/families.json is Table 1 of the post (45 versions), copied unchanged. These are reported scores
from different papers, so every number here is a description of the table, not a controlled finding.
"""

import _path  # noqa: F401

import json

from vlmfam.report import DATA, Report

table = json.loads((DATA / "families.json").read_text())
ROWS, BENCH = table["rows"], table["benchmarks"]
r = Report("08_cross_family")


def score(row, bench):
    return row["b"][BENCH.index(bench)]


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):  # average ranks for ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs, ys):
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return cov / var


def month(row):
    y, m = row["rel"].split("-")
    return (int(y) - 2023) * 12 + int(m)


r(f"Table 1 of the post: {len(ROWS)} versions. Image input: "
  + ", ".join(f"{k} {sum(1 for x in ROWS if x['prep'] == k)}" for k in ("fixed", "tiling", "native", "xattn")))

# claim 1: DocVQA >= 90 needs tiling or native resolution
r("")
r("Claim (Strong): 'every model at or above 90 on DocVQA uses tiling or native resolution'")
hi = [x for x in ROWS if score(x, "DocVQA") is not None and score(x, "DocVQA") >= 90]
for kind in ("fixed", "tiling", "native", "xattn"):
    names = [f"{x['name']} {score(x, 'DocVQA')}" for x in hi if x["prep"] == kind]
    r(f"  {kind:<7} {len(names):>2}  {', '.join(names)}")
fixed = sorted((score(x, "DocVQA"), x["name"]) for x in ROWS if x["prep"] == "fixed" and score(x, "DocVQA"))
r(f"  best fixed-size DocVQA: {', '.join(f'{n} {s}' for s, n in fixed[::-1][:3])}")
r("  The xattn row is Llama 3.2 Vision, which also tiles its input (up to 4 tiles of 560 px), so the claim holds.")

# claim 2: MMMU tracks the language model (hypothesis) and the date confound
r("")
r("Hypothesis: 'MMMU tracks the language model more than anything visual'. Spearman rank correlations:")
rows = []
for bench in BENCH:
    pts = [(x["llm_b"], score(x, bench), month(x), x["maxtok"]) for x in ROWS
           if x["llm_b"] and score(x, bench) is not None]
    tok = [(t, s) for _, s, _, t in pts if t]
    rows.append([bench, len(pts), f"{spearman([p[0] for p in pts], [p[1] for p in pts]):+.2f}",
                 f"{spearman([p[2] for p in pts], [p[1] for p in pts]):+.2f}",
                 f"{spearman([t for t, _ in tok], [s for _, s in tok]):+.2f} (n={len(tok)})"])
r.table(["benchmark", "models", "vs LLM size", "vs release month", "vs max visual tokens"], rows)
both = [x for x in ROWS if x["llm_b"]]
r(f"  LLM size vs release month over all {len(both)} rows with a size: "
  f"{spearman([x['llm_b'] for x in both], [month(x) for x in both]):+.2f}")
r("  Every benchmark ranks more closely with release month than with LLM size, and the visual token budget also")
r("  grows over time. Newer models differ in data, RL and resolution at once, so no single column isolates the")
r("  language model. This is why the post files the MMMU claim as a hypothesis.")

# claim 3: preference / RL stages and math
r("")
r("Strong (within-paper) claim on RL and math; the cross-family view only:")
for flag in (True, False):
    xs = [score(x, "MathVista") for x in ROWS if x["rlflag"] == flag and score(x, "MathVista") is not None]
    ms = [month(x) for x in ROWS if x["rlflag"] == flag and score(x, "MathVista") is not None]
    r(f"  {'with' if flag else 'without'} a preference or RL stage: n={len(xs)}, mean MathVista "
      f"{sum(xs) / len(xs):.1f}, mean release {2023 + (sum(ms) / len(ms) - 1) // 12:.0f}-"
      f"{(sum(ms) / len(ms) - 1) % 12 + 1:02.0f}")
mean_month = {f: [month(x) for x in ROWS if x["rlflag"] == f and score(x, "MathVista") is not None]
              for f in (True, False)}
gap = sum(mean_month[True]) / len(mean_month[True]) - sum(mean_month[False]) / len(mean_month[False])
r(f"  The RL rows are also {gap:.0f} months newer on average, so this gap is not evidence by itself; the controlled rows in the")
r("  post (InternVL3 MPO +5.0, InternVL3.5 Cascade RL +6.7) are.")
r.save({"docvqa_90_plus": [[x["name"], x["prep"], score(x, "DocVQA")] for x in hi], "spearman": rows})
