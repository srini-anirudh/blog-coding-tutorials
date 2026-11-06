"""Sections 2 and 3: static batching versus continuous (iteration-level) batching, same requests, same model.

Every request is in the queue at t = 0. Output lengths are log-uniform between 8 and 512 tokens, so lengths
differ by up to 64x. Static batching admits a group of B and runs it until the longest member is done; the
finished members keep computing. Continuous batching refills a slot on the very next iteration.
Both use the same paged forward pass, so the difference is only the scheduling decision.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from servekit.engine import Engine  # noqa: E402
from servekit.metrics import Report, device_and_dtype, output_lengths, random_prompts, summarize  # noqa: E402
from servekit.model import load_model  # noqa: E402
from servekit.scheduler import SchedulerConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--requests", type=int, default=256)
    ap.add_argument("--max-out", type=int, default=512)
    args = ap.parse_args()
    dev, dtype = device_and_dtype()
    model = load_model(args.model, dtype, dev)
    vocab = model.config.vocab_size
    prompts = random_prompts(args.requests, 32, 512, vocab, seed=0)
    outs = output_lengths(args.requests, 8, args.max_out, seed=0)
    rep = Report("03_batching.txt")
    rep(f"model {args.model}; {args.requests} requests at t=0; prompts 32-512 tokens; outputs 8-{args.max_out} tokens"
        f" (mean {sum(outs) / len(outs):.0f}); greedy; block size 16")
    rows, effective = [], {}
    for mode, slots in (("static", 32), ("continuous", 32), ("static", 128), ("continuous", 128)):
        cfg = SchedulerConfig(max_num_seqs=slots, max_num_batched_tokens=slots * 512 if mode == "static" else 4096,
                              chunked_prefill=mode != "static", mode=mode)
        eng = Engine(model, num_blocks=16000, sched=cfg)
        wall = eng.run([(0.0, p, n) for p, n in zip(prompts, outs)])
        s = summarize(eng.finished, wall)
        dec = [st for st in eng.stats if st.num_decodes]
        eff = sum(st.num_useful_decodes for st in dec) / len(dec)
        rows.append({"mode": mode, "slots": slots, "wall_s": wall, "out_tok/s": s["out_tok_per_s"],
                     "mean_latency_s": s["latency_mean"], "ttft_p50_s": s["ttft_p50"], "steps": len(eng.stats),
                     "useful_batch": eff})
        if mode == "static" and slots == 32:
            g = []
            for st in eng.stats[1:]:  # stop when the second group is admitted
                if not st.num_decodes or (g and st.num_useful_decodes > g[-1]):
                    break
                g.append(st.num_useful_decodes)
            effective = {"first_group_useful_decodes_per_step": g}
        preempted = eng.sched.num_preemptions
        assert preempted == 0, "the pool is large enough that no request is preempted"
        del eng
        gc.collect()
        torch.cuda.empty_cache()
    rep.table(rows, {"wall_s": "{:.1f}", "out_tok/s": "{:.0f}", "mean_latency_s": "{:.2f}", "ttft_p50_s": "{:.2f}",
                     "useful_batch": "{:.1f}"})
    rep("useful_batch = mean number of sequences per decode step that still wanted a token.")
    g = effective["first_group_useful_decodes_per_step"]
    rep("Static, 32 slots, first group: sequences still wanting tokens at decode step "
        + ", ".join(f"{i}: {g[i]}" for i in (0, 8, 16, 32, 64, 128, 256, len(g) - 1) if i < len(g)))
    rep.save({"rows": rows, **effective})


if __name__ == "__main__":
    main()
