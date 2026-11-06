"""Score the saved replies of 10_run_models.py. CPU only, seconds to run.

For each model on MMStar: accuracy under three extractors and two prompts (the parser and the
prompt are part of the metric), tiling on versus off (input settings), the blind test (MMStar's
multi-modal gain and leakage), CircularEval and the letter the model picks (position bias),
bootstrap intervals, and a paired comparison of the two models. For POPE: accuracy, precision,
recall, F1 and the yes-ratio in each setting.

    python labs/measuring-what-a-vlm-can-do/scripts/11_analyse.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

import torch  # noqa: E402

from benchlab.mc import EXTRACTORS, LETTERS, MCItem, extract  # noqa: E402
from benchlab.metrics import pope_stats, yes_no  # noqa: E402
from benchlab.report import Report  # noqa: E402
from benchlab.stats import bootstrap_ci, multimodal_gain_leakage, paired_bootstrap, standard_error  # noqa: E402

RAW = LAB / "results" / "raw"


def items_from(raw: dict) -> list[MCItem]:
    """Extractors need the number of options and, for content matching, the option texts.

    The raw file stores gold letters and option counts. Option texts come from the dataset, but
    only the 'content' extractor needs them, so we reload MMStar for it when it is cached.
    """
    from benchlab.runner import load_mmstar

    rows, _ = load_mmstar()
    by_index = {r.index: r.item for r in rows}
    return [by_index[i] for i in raw["index"]]


def score(replies: list[str], items: list[MCItem], how: str, k: int = 0) -> torch.Tensor:
    from benchlab.mc import rotate

    return torch.tensor([extract(rep, rotate(it, k), how) == rotate(it, k).gold_letter
                         for rep, it in zip(replies, items)], dtype=torch.float)


def pct(x: float) -> str:
    return f"{100 * x:5.1f}"


def mmstar(r: Report, name: str, raw: dict, summary: dict) -> torch.Tensor:
    items = items_from(raw)
    rep = raw["replies"]
    n = len(items)
    r(f"\n### MMStar, {raw['model']} (n = {n}, skipped {raw['skipped']}; replies from: {raw['hardware']})")
    r(f"mean image tokens per prompt: tiling on {raw['image_tokens']['fmt']:.0f}, "
      f"tiling off {raw['image_tokens']['fmt_notile']:.0f}")

    r("\naccuracy (%) by prompt and extractor")
    r(f"{'condition':34s} " + " ".join(f"{e:>10s}" for e in EXTRACTORS))
    acc = {}
    for cond, label in [("fmt", "image, format hint"), ("bare", "image, no hint"),
                        ("fmt_notile", "image, hint, tiling off"), ("blind", "no image (S_wv)"),
                        ("llm", f"base LLM {raw['base_llm'].split('/')[-1]} (S_t)")]:
        acc[cond] = {e: score(rep[cond], items, e) for e in EXTRACTORS}
        r(f"{label:34s} " + " ".join(f"{pct(acc[cond][e].mean().item()):>10s}" for e in EXTRACTORS))

    main = acc["fmt"]["content"]
    lo, hi = bootstrap_ci(main)
    p = main.mean().item()
    r(f"\nheadline (image, hint, content extractor): {pct(p)}%  bootstrap 95% CI [{pct(lo)}, {pct(hi)}]"
      f"  formula SE {100 * standard_error(p, n):.1f} pt")

    mgml = multimodal_gain_leakage(p, acc["blind"]["content"].mean().item(), acc["llm"]["content"].mean().item())
    r(f"blind test: S_v {pct(p)}  S_wv {pct(acc['blind']['content'].mean().item())}  "
      f"S_t {pct(acc['llm']['content'].mean().item())}  ->  MG {100 * mgml['MG']:.1f} pt, ML {100 * mgml['ML']:.1f} pt")
    gain = paired_bootstrap(acc["fmt"]["content"], acc["blind"]["content"])
    r(f"  image versus no image, paired bootstrap: {100 * gain['diff']:+.1f} pt [{100 * gain['lo']:+.1f}, {100 * gain['hi']:+.1f}]")

    r("\nper category (content extractor): image / no image")
    cats = raw["category"]
    for c in sorted(set(cats)):
        m = torch.tensor([x == c for x in cats])
        r(f"  {c:24s} {pct(acc['fmt']['content'][m].mean().item())} / {pct(acc['blind']['content'][m].mean().item())}")

    # CircularEval over items with 4 options (MMStar's are all four-option after parsing).
    passes = [score(rep["fmt"], items, "content")] + [score(rep[f"rot{k}"], items, "content", k) for k in (1, 2, 3)]
    four = torch.tensor([len(it.options) == 4 for it in items])
    stack = torch.stack(passes)[:, four]
    circ = stack.min(0).values
    r(f"\nCircularEval ({int(four.sum())} four-option items): single pass {pct(stack[0].mean().item())}%,"
      f" mean over the 4 rotations {pct(stack.mean().item())}%, CircularEval {pct(circ.mean().item())}%")

    picks = collections.Counter(extract(x, it, "content") for x, it in zip(rep["fmt"], items))
    gold = collections.Counter(raw["gold"])
    r("letter picked (fmt, content extractor): " + ", ".join(f"{k}: {picks.get(k, 0)}" for k in LETTERS[:4])
      + f", unparsed: {picks.get(None, 0)}")
    r("gold letters in MMStar:                 " + ", ".join(f"{k}: {gold.get(k, 0)}" for k in LETTERS[:4])
      + "  (one constant letter scores: " + ", ".join(f"{k} {pct(gold.get(k, 0) / n).strip()}%" for k in LETTERS[:4]) + ")")

    summary[name] = {
        "n": n, "headline": p, "ci95": [lo, hi],
        "accuracy": {c: {e: v.mean().item() for e, v in d.items()} for c, d in acc.items()},
        "MG": mgml["MG"], "ML": mgml["ML"], "circular": circ.mean().item(), "single_pass": stack[0].mean().item(),
        "picks": {str(k): v for k, v in picks.items()}, "image_tokens": raw["image_tokens"],
        "hardware": raw["hardware"], "gpu_seconds": raw["total_seconds"],
    }
    return main


def pope(r: Report, name: str, raw: dict, summary: dict) -> None:
    r(f"\n### POPE, {raw['model']} (replies from: {raw['hardware']})")
    r(f"{'setting':12s} {'n':>5s} {'acc':>6s} {'prec':>6s} {'recall':>6s} {'F1':>6s} {'yes-ratio':>9s} {'unparsed':>8s}")
    summary[name] = {}
    for s, replies in raw["replies"].items():
        preds = [yes_no(x) for x in replies]
        st = pope_stats(preds, raw["gold"][s])
        st["unparsed"] = sum(p is None for p in preds)
        summary[name][s] = st
        r(f"{s:12s} {st['n']:5d} {pct(st['accuracy']):>6s} {pct(st['precision']):>6s} {pct(st['recall']):>6s} "
          f"{pct(st['f1']):>6s} {st['yes_ratio']:9.3f} {st['unparsed']:8d}")


def main():
    r = Report("11_analyse")
    files = sorted(p for p in RAW.glob("*.json") if "limit" not in p.name)
    if not files:
        r("no raw replies yet: run 10_run_models.py first")
        return
    from benchlab.report import command_line, hardware

    summary = {"hardware": f"scored on {hardware()}; replies generated on the hardware named per model",
               "command": command_line()}
    correct = {}
    for f in files:
        raw = json.loads(f.read_text())
        if f.name.startswith("mmstar"):
            correct[f.stem] = mmstar(r, f.stem, raw, summary)
        else:
            pope(r, f.stem, raw, summary)
    if len(correct) == 2:
        (na, a), (nb, b) = sorted(correct.items(), key=lambda kv: -kv[1].mean().item())
        d = paired_bootstrap(a, b)
        r(f"\n### Two models on the same 1,500 items: {na} minus {nb}")
        r(f"difference {100 * d['diff']:+.1f} pt, paired bootstrap 95% CI [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}],"
          f" share of resamples with difference <= 0: {d['p_le_0']:.3f}")
        summary["paired"] = {"a": na, "b": nb, **d}
    (LAB / "results" / "11_analyse.json").write_text(json.dumps(summary, indent=1))
    r.save()


if __name__ == "__main__":
    main()
