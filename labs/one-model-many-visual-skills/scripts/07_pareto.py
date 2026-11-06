"""The Pareto view of mixture tuning, with real runs (post Figure 10 uses an illustrative model).

Ten random mixtures over VQA, chat, captions and text replay, each trained with the same budget,
once without and once with the format prompt. Two capabilities compete when the formats clash:
short-answer accuracy (VQA exact) and conversational style (share of long answers to unmarked
questions). We find the mixtures that no other mixture beats on both, then pick one point by a
weighted sum and one by a constraint: best VQA with long rate >= 0.9 and text >= base - 0.01.

    python labs/one-model-many-visual-skills/scripts/07_pareto.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillmix.evaluate import eval_text  # noqa: E402
from skillmix.mixture import pareto_front, pick_constrained, pick_weighted  # noqa: E402
from skillmix.runs import base_model, sweep, write_result  # noqa: E402
from skillmix.train import RunConfig  # noqa: E402

SOURCES = ("vqa", "chat", "caption", "text")
N_MIXES = 10


def mixtures(n: int, seed: int = 7) -> list[dict]:
    """Random points on the simplex (a flat Dirichlet)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        g = [rng.gammavariate(1.0, 1.0) for _ in SOURCES]
        out.append({k: round(v / sum(g), 3) for k, v in zip(SOURCES, g)})
    return out


def main() -> None:
    base_text = eval_text(base_model())["text_exact"]
    mixes = mixtures(N_MIXES)
    jobs = []
    for fmt in (False, True):
        suite = [("eval_vqa", {"fmt": fmt}), ("eval_chat", {}), ("eval_text", {})]
        jobs += [(f"m{i:02d} fmt={'on' if fmt else 'off'}", RunConfig(mixture=m, fmt=fmt), suite)
                 for i, m in enumerate(mixes)]
    res = sweep(jobs)
    out, lines = {}, []
    for fmt in ("off", "on"):
        names = [n for n in res if n.endswith(f"fmt={fmt}")]
        pts = [(res[n]["vqa_exact"], res[n]["chat_long_rate"]) for n in names]
        front = pareto_front(pts)
        lines += [f"format prompt {fmt}",
                  f"{'run':<12} {'vqa':>5} {'chat':>5} {'cap':>5} {'text':>5} | {'VQA':>6} {'long':>6} {'text':>6}  front"]
        for i, n in enumerate(names):
            m, r = mixes[i], res[n]
            lines.append(f"{n:<12} {m['vqa']:>5.2f} {m['chat']:>5.2f} {m['caption']:>5.2f} {m['text']:>5.2f} | "
                         f"{r['vqa_exact']:>6.3f} {r['chat_long_rate']:>6.3f} {r['text_exact']:>6.3f}  "
                         f"{'*' if i in front else ''}")
        picks = {f"weighted {w}": names[pick_weighted(pts, w)] for w in ((0.8, 0.2), (0.5, 0.5), (0.2, 0.8))}
        ok = [i for i, n in enumerate(names) if res[n]["text_exact"] >= base_text - 0.01]
        sub = pick_constrained([pts[i] for i in ok], maximise=0, floors={1: 0.9})
        picks["constraint: max VQA, long >= 0.9, text >= base - 0.01"] = names[ok[sub]] if sub is not None else None
        lines += [f"front: {', '.join(names[i] for i in front)}", "picks:"] + [f"  {k}: {v}" for k, v in picks.items()]
        lines.append("")
        out[fmt] = {"front": [names[i] for i in front], "picks": picks}
    lines.append(f"base text score {base_text:.3f}. * = on the Pareto front of (VQA exact, long rate).")
    text = "\n".join(lines)
    print(text)
    write_result("07_pareto", {"mixtures": {f"m{i:02d}": m for i, m in enumerate(mixes)}, "runs": res, **out,
                               "base_text_exact": base_text}, text)


if __name__ == "__main__":
    main()
