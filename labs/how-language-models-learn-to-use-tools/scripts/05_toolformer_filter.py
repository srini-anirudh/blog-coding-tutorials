"""Does the call pay for itself? Toolformer's loss-differential filter, on small pretrained models.

For each call site there are three candidate calls: a useful one, one whose own text already gives the
answer away, and an irrelevant one. The script prints L(empty), L(call, no result) and L(call, result)
over the next five tokens, and how many calls each threshold tau_f keeps, with and without the min.

    python labs/how-language-models-learn-to-use-tools/scripts/05_toolformer_filter.py \
        --models openai-community/gpt2 HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from toollab.hw import header  # noqa: E402
from toollab.toolformer import corpus, filter_call, toolformer_weights  # noqa: E402

TAUS = (0.5, 1.0, 2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["openai-community/gpt2"])
    ap.add_argument("--out", default=str(LAB / "results" / "toolformer_filter.json"))
    args = ap.parse_args()
    hd = header()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    print("weights w_0..w_4:", [round(float(w), 4) for w in toolformer_weights()])
    sites = corpus()
    print(f"{len(sites)} call sites x 3 candidate calls\n")
    out = {**hd, "models": {}}
    for name in args.models:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        rows = []
        for s in sites:
            for kind, (call, result) in s["calls"].items():
                rows.append({"tool": s["tool"], "kind": kind, "call": call, "result": result,
                             **filter_call(model, tok, s["before"], s["after"], call, result)})
        n_params = sum(p.numel() for p in model.parameters())
        print(f"== {name} ({n_params / 1e6:.0f}M parameters)")
        print(f"{'':<26}{'':>27} | {'kept at tau_f':^20} | {'kept without the min':^20}")
        print(f"{'tool':<11}{'candidate':<15}{'L(empty)':>9}{'L(call)':>9}{'L(+res)':>9} | "
              + "".join(f"{t:>7}" for t in TAUS) + " | " + "".join(f"{t:>7}" for t in TAUS))
        groups = defaultdict(list)
        for r in rows:
            groups[(r["tool"], r["kind"])].append(r)
        summary = {}
        for (tool, kind), rs in groups.items():
            m = lambda k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
            kept = [sum(r["gain"] >= t for r in rs) for t in TAUS]
            kept_nomin = [sum(r["gain_without_min"] >= t for r in rs) for t in TAUS]
            summary[f"{tool}/{kind}"] = {"n": len(rs), "mean_L_empty": round(m("L_empty"), 3),
                                         "mean_L_call_no_result": round(m("L_call_no_result"), 3),
                                         "mean_L_plus": round(m("L_plus"), 3),
                                         "kept": dict(zip(map(str, TAUS), kept)),
                                         "kept_without_min": dict(zip(map(str, TAUS), kept_nomin))}
            print(f"{tool:<11}{kind:<15}{m('L_empty'):>9.2f}{m('L_call_no_result'):>9.2f}{m('L_plus'):>9.2f} | "
                  + "".join(f"{f'{k}/{len(rs)}':>7}" for k in kept)
                  + " | " + "".join(f"{f'{k}/{len(rs)}':>7}" for k in kept_nomin))
        print()
        out["models"][name] = {"parameters": n_params, "summary": summary, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
