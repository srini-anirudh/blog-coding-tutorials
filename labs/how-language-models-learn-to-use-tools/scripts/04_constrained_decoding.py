"""Syntax is free, semantics is not. Mask the vocabulary with the schema and see what it fixes.

Part 1 is exact. A two-step toy distribution shows that per-step masking does not sample from
p(y | y is valid), that greedy masked decoding can return the wrong valid string, and that weighting each
step by alpha(prefix) recovers the exact conditional.

Part 2 decodes the trained tiny models (from 03_tool_diversity.py) on tools they never saw, three ways:
no mask, a schema mask (tool names, field names and punctuation forced; any value allowed), and a typed
mask (each value must also have its field's type). It reports parse rate and exact-call rate.

    python labs/how-language-models-learn-to-use-tools/scripts/04_constrained_decoding.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from toollab.hw import header  # noqa: E402
from toollab.evaluate import eval_set, score, summarise  # noqa: E402
from toollab.grammar import call_mask, local_vs_global  # noqa: E402
from toollab.world import Vocab, make_tools, split_names  # noqa: E402

# The model would rather call tool A, but the value it wants to give A has the wrong type.
TOY = {
    (): {"A": 0.6, "B": 0.4},
    ("A",): {"x": 0.9, "y": 0.1},  # A wants a day; "x" is a city: invalid
    ("B",): {"x": 0.95, "z": 0.05},  # B wants a city; "x" is valid
}
VALID = {("A", "y", "$"), ("B", "x", "$")}


def toy_next(prefix):
    if len(prefix) == 2:
        return {"$": 1.0}
    return TOY[prefix]


def part1() -> dict:
    res = local_vs_global(toy_next, lambda s: s in VALID)
    print("Part 1: exact distributions over the two valid strings")
    print(f"{'string':<8}{'p(y|valid)':>12}{'local mask':>12}{'alpha-corrected':>17}")
    for s in sorted(res["exact"]):
        print(f"{''.join(s[:2]):<8}{res['exact'][s]:>12.4f}{res['local'][s]:>12.4f}{res['alpha_corrected'][s]:>17.4f}")
    print("alpha after the first token:", {k[0]: round(v, 3) for k, v in res["alpha"].items()})
    print(f"KL(exact || local) = {res['kl_exact_to_local']:.4f} nats")
    greedy_local = max(res["local"], key=res["local"].get)
    greedy_exact = max(res["exact"], key=res["exact"].get)
    print(f"greedy with a local mask picks {''.join(greedy_local[:2])}; the most likely valid string is "
          f"{''.join(greedy_exact[:2])}\n")
    return {"exact": {"".join(k[:2]): v for k, v in res["exact"].items()},
            "local": {"".join(k[:2]): v for k, v in res["local"].items()},
            "alpha_corrected": {"".join(k[:2]): v for k, v in res["alpha_corrected"].items()},
            "kl": res["kl_exact_to_local"]}


def part2(n_eval: int) -> dict:
    from toollab.model import decode, load

    ckpts = sorted((LAB / "runs").glob("tools_N*.pt"), key=lambda p: int(p.stem.split("N")[1]))
    if not ckpts:
        print("Part 2 skipped: run scripts/03_tool_diversity.py first.")
        return {}
    vocab = Vocab()
    train_names, heldout = split_names(seed=0)
    unseen = make_tools(300, heldout, random.Random(7))
    exs = eval_set(unseen, n_eval, seed=303)
    print(f"Part 2: {n_eval} requests over unseen tools")
    print(f"{'N':>6}  {'decoding':<10}{'parse':>7}{'exact':>7}{'select':>8}{'args':>7}")
    out = {}
    for ck in ckpts:
        model, meta = load(ck)
        n = meta["n_tools"]
        train_set = {t.name_str for t in make_tools(n, train_names, random.Random(meta["seed"] + n))}
        out[n] = {}
        for label, mk in (("none", None), ("schema", lambda ex: call_mask(ex.tools)),
                          ("typed", lambda ex: call_mask(ex.tools, typed=True))):
            rows = [score(ex, decode(model, vocab, ex, mask=mk(ex) if mk else None), train_set) for ex in exs]
            s = summarise(rows)
            out[n][label] = s
            print(f"{n:>6}  {label:<10}{s['parse']:>7.3f}{s['exact']:>7.3f}{s['select_given_call'] or 0:>8.3f}"
                  f"{s['args_given_tool'] or 0:>7.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--out", default=str(LAB / "results" / "constrained_decoding.json"))
    args = ap.parse_args()
    hd = header()
    res = {**hd, "toy": part1(), "tiny_models_unseen_tools": part2(args.n_eval)}
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
