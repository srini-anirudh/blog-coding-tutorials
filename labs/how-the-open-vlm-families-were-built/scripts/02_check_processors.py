"""Check the from-scratch rules in vlmfam/tokens.py against the real Hugging Face processors.

Network: downloads processor files only (no weights). For every repo and every test image size, the processor's
token count must equal the rule's. Writes results/02_check_processors.{txt,json}.
"""

import _path  # noqa: F401

from vlmfam import processors as P
from vlmfam.report import Report

r = Report("02_check_processors")
data, total, agree = [], 0, 0
for label, repo in P.PROCESSORS:
    probe = P.Probe(label, repo)
    r(f"{label} | {repo} | {probe.how}")
    r(f"  settings: {probe.settings()}")
    row = {"rule": label, "repo": repo, "how": probe.how, "sizes": []}
    cells = []
    for w, h in P.SIZES:
        m, k = probe.measure(w, h), probe.rule(w, h)
        total += 1
        agree += m == k
        row["sizes"].append({"width": w, "height": h, "processor": m, "rule": k})
        cells.append(f"{w}x{h}={m}" + ("" if m == k else f" (rule {k})"))
    r("  " + ", ".join(cells))
    data.append(row)
r("")
r(f"{agree} of {total} (repo, size) pairs: rule == processor")
r.save(data)
assert agree == total, "a rule disagrees with its processor"
