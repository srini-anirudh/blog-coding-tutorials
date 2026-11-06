"""How document reading is scored: ANLS and its threshold, the OmniDocBench v1.5 overall formula checked against the
component scores the post quotes, and binary unit tests (olmOCR-Bench style) against edit distance.

Post section: "How document reading is measured", Figure 8, and the nuances on edit distance and repetition.
The OmniDocBench numbers below are copied from the post's Figure 8 (sources named there); nothing is re-run.
"""

import _common
from docread.metrics import (anls, anls_single, check_absent, check_no_repetition, check_order, check_present,
                             longest_repeat_run, normalized_edit_distance, omnidocbench_overall)

log = _common.Report("07_scoring")
data = {}

log("ANLS for one question with gold answer '1,500.00':")
for pred in ("1,500.00", "1,580.00", "1500.00", "1,500", "15", "2,750.00"):
    nl = normalized_edit_distance(pred, "1,500.00")
    log(f"  {pred:<10} NL {nl:.3f}  ANLS {anls_single(pred, ['1,500.00']):.3f}")
data["anls_examples"] = {p: anls_single(p, ["1,500.00"]) for p in ("1,500.00", "1,580.00", "15")}
log("A one-digit misread of an amount keeps 0.875 of the credit. Exact match would give it 0.")
preds = ["1,580.00", "Northwind", "2026-03-14", "INV-2014"]
golds = [["1,500.00"], ["Northwind Traders", "Northwind"], ["2026-03-14", "14 March 2026"], ["INV-2041"]]
log(f"Four answers, two of them wrong by one or two characters: ANLS {anls(preds, golds):.3f}, "
    f"exact match {sum(p in g for p, g in zip(preds, golds)) / 4:.3f}\n")

# name: (reported overall, text edit, formula CDM, table TEDS), from Figure 8 of the post.
FIG8 = {"Marker-1.8.2": (71.30, 0.206, 76.66, 57.88), "MinerU2-pipeline": (75.51, 0.209, 76.55, 70.90),
        "PP-StructureV3": (86.73, 0.073, 85.79, 81.68), "GPT-4o": (75.02, 0.217, 79.70, 67.07),
        "InternVL3.5-241B": (82.67, 0.142, 87.23, 75.00), "Qwen2.5-VL-72B": (87.02, 0.094, 88.27, 82.15),
        "Gemini-2.5 Pro": (88.03, 0.075, 85.82, 85.71), "Qwen3-VL-235B": (89.15, 0.069, 88.14, 86.21),
        "Mistral OCR": (78.83, 0.164, 82.84, 70.03), "olmOCR-7B": (81.79, 0.096, 86.04, 68.92),
        "DeepSeek-OCR (9 crops)": (87.36, 0.073, 84.14, 85.25), "dots.ocr": (88.41, 0.048, 83.22, 86.78),
        "MinerU2.5": (90.67, 0.047, 88.46, 88.22), "DeepSeek-OCR 2": (91.09, 0.048, 90.31, 87.75),
        "PaddleOCR-VL": (92.86, 0.035, 91.22, 90.89)}
log("OmniDocBench v1.5 overall = [(1 - TextEdit) x 100 + Table TEDS + Formula CDM] / 3, recomputed from components:")
log(f"  {'system':<24} {'reported':>8} {'recomputed':>10} {'diff':>6}")
worst = 0.0
for name, (overall, text, cdm, teds) in FIG8.items():
    mine = omnidocbench_overall(text, teds, cdm)
    worst = max(worst, abs(mine - overall))
    log(f"  {name:<24} {overall:>8.2f} {mine:>10.2f} {mine - overall:>+6.2f}")
data["omnidocbench_max_abs_diff"] = worst
log(f"Largest gap {worst:.2f} points: the components are rounded (text edit to 3 decimals), and the formula reproduces every reported overall.\n")

reference = "Results\nTable 2 lists the edit distance by page type.\nFigure 3: error by page type.\nScanned pages score 0.118."
outputs = {
    "faithful": reference,
    "caption moved after the paragraph": "Results\nTable 2 lists the edit distance by page type.\nScanned pages score 0.118.\nFigure 3: error by page type.",
    "running header kept": "Page 7 | Doc AI Survey\n" + reference,
    "one digit wrong": reference.replace("0.118", "0.178"),
    "stuck in a loop": reference + "\nScanned pages score 0.118." * 6,
}
tests = [("present: 'Scanned pages score 0.118.'", lambda o: check_present(o, "Scanned pages score 0.118.")),
         ("absent: running header", lambda o: check_absent(o, "Doc AI Survey")),
         ("order: heading before table text", lambda o: check_order(o, "Results", "Table 2 lists")),
         ("no repetition (max run 3)", lambda o: check_no_repetition(o))]
log("Edit distance against binary unit tests on five outputs of one small page:")
log(f"  {'output':<36} {'edit dist':>9} {'tests passed':>12} {'longest repeat':>15}")
data["unit_tests"] = {}
for name, out in outputs.items():
    passed = [fn(out) for _, fn in tests]
    ed = normalized_edit_distance(out, reference)
    data["unit_tests"][name] = {"edit_distance": ed, "passed": passed}
    log(f"  {name:<36} {ed:>9.3f} {sum(passed):>8} / {len(tests)} {longest_repeat_run(out):>15}")
log("  tests: " + "; ".join(t for t, _ in tests))
log("Moving a caption costs edit distance but breaks nothing a reader needs; a one-digit error costs almost no edit")
log("distance and changes a fact. The unit tests see the second only if someone wrote a test for that number.")
log.save(data)
