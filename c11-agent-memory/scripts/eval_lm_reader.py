"""Same contexts, real reader: Qwen2.5-0.5B-Instruct answers the fact probes.

The rule reader is perfect by construction, so it cannot show what a long context does
to a model. This script swaps only the reader, using the C10 HF client.

    python c11-agent-memory/scripts/eval_lm_reader.py --seeds 3 [--device mps]

Appends progress to results/lm_reader.json after every script. Takes several minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from agentmem import count_tokens  # noqa: E402
from agentmem.bench.facts_eval import FactsEvaluator, ProbeResult, is_correct, summarise  # noqa: E402
from agentmem.bench.facts_world import make_script  # noqa: E402
import agentmem.integration  # noqa: E402,F401  (puts ../c10-agent-harness on sys.path)
from harness import HFLocalModel  # noqa: E402

ASK = "\n\nReply with only the value, nothing else. If the context does not say, reply: unknown."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()
    torch.set_num_threads(2)
    model = HFLocalModel(args.model, device=args.device, max_new_tokens=16)
    results: list[ProbeResult] = []
    samples = []
    t0 = time.perf_counter()
    long_ev = FactsEvaluator(budget=8192)  # the whole transcript fits: the "just send it all" case
    for seed in range(args.seeds):
        script = make_script(seed)
        ev = FactsEvaluator(args.budget)
        for (probe, ctx), (_, ctx_long) in zip(ev.contexts(script), long_ev.contexts(script)):
            cases = dict(ctx)
            cases["full_transcript_unbounded"] = ctx_long["full_transcript"]
            for cond, text in cases.items():
                reply = model.complete([{"role": "user", "content": text + ASK}], []).text
                results.append(ProbeResult(probe, cond, reply, is_correct(reply, probe.gold), count_tokens(text)))
                if len(samples) < 40 and cond == "full_transcript_unbounded":
                    samples.append({"q": f"{probe.attribute} for {probe.entity}", "gold": probe.gold, "kind": probe.kind,
                                    "reply": reply[:60]})
        rows = summarise(results)
        out = {"model": args.model, "device": model.device, "budget": args.budget, "seeds_done": seed + 1,
               "wall_s": round(time.perf_counter() - t0), "rows": rows, "samples": samples}
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / "lm_reader.json").write_text(json.dumps(out, indent=1, sort_keys=True))
        print(f"seed {seed} done, {time.perf_counter() - t0:.0f}s", flush=True)
    kinds = ["stated", "updated", "rumor", "deleted", "today", "after_today", "unseen"]
    print(f"{'condition':<28}{'accuracy':>9}{'tokens':>8}" + "".join(f"{k:>12}" for k in kinds))
    for cond, r in summarise(results).items():
        print(f"{cond:<28}{r['accuracy']:>9.3f}{r['tokens']:>8.0f}" + "".join(f"{r.get(k, float('nan')):>12.2f}" for k in kinds))


if __name__ == "__main__":
    main()
