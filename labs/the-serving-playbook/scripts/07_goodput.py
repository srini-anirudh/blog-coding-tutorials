"""Section 23: sweep offered load. Throughput saturates; goodput (requests per second that met both SLOs)
peaks and then falls. SLO-aware admission (shed requests that queued too long) keeps part of it.

Requests arrive as a Poisson process. Prompts 128-1,024 tokens, outputs log-uniform 32-256 tokens.
A request meets its SLO if TTFT <= --ttft-slo and its mean TPOT <= --tpot-slo. Each load level is one run
of about --seconds of arrivals, so the percentiles carry sampling noise.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from servekit.engine import Engine  # noqa: E402
from servekit.metrics import (Report, device_and_dtype, output_lengths, poisson_arrivals,  # noqa: E402
                              random_prompts, summarize)
from servekit.model import load_model  # noqa: E402
from servekit.scheduler import SchedulerConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--rates", default="2,4,8,12,16,24,32,48")
    ap.add_argument("--eager", action="store_true", help="no CUDA graph for pure-decode iterations")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--ttft-slo", type=float, default=1.0)
    ap.add_argument("--tpot-slo", type=float, default=0.08)
    args = ap.parse_args()
    dev, dtype = device_and_dtype()
    model = load_model(args.model, dtype, dev)
    v = model.config.vocab_size
    rep = Report("07_goodput.txt")
    rep(f"model {args.model}; Poisson arrivals for about {args.seconds:.0f} s per load level; SLO: TTFT <= "
        f"{args.ttft_slo * 1e3:.0f} ms and mean TPOT <= {args.tpot_slo * 1e3:.0f} ms; max 128 sequences, token budget 2048; "
        f"{'eager' if args.eager else 'CUDA-graph'} decode iterations")
    warm = Engine(model, num_blocks=8000)
    warm.run([(0.0, p, 16) for p in random_prompts(8, 128, 1024, v, seed=99)])
    rows = []
    for rate in [float(r) for r in args.rates.split(",")]:
        n = max(20, int(rate * args.seconds))
        prompts = random_prompts(n, 128, 1024, v, seed=int(rate))
        outs = output_lengths(n, 32, 256, seed=int(rate))
        arr = poisson_arrivals(n, rate, seed=int(rate))
        for shed in (None, args.ttft_slo):
            eng = Engine(model, num_blocks=12000, shed_after=shed, cuda_graph=not args.eager and dev == "cuda", precapture=True,
                         max_model_len=2048,
                         sched=SchedulerConfig(max_num_seqs=128, max_num_batched_tokens=2048))
            wall = eng.run(list(zip(arr, prompts, outs)))
            s = summarize(eng.finished, wall, args.ttft_slo, args.tpot_slo)
            rows.append({"offered_req/s": rate, "admission": "shed" if shed else "accept all",
                         "throughput_req/s": s["req_per_s"], "out_tok/s": s["out_tok_per_s"],
                         "goodput_req/s": s["goodput_req_per_s"],
                         "slo_met": s["slo_attainment"] * len(eng.finished) / n, "shed": len(eng.shed),
                         "ttft_p99_ms": s["ttft_p99"] * 1e3, "tpot_p99_ms": s["tpot_p99"] * 1e3})
            print(f"  rate {rate:g} {rows[-1]['admission']}: goodput {rows[-1]['goodput_req/s']:.2f} req/s", flush=True)
            del eng
            gc.collect()
            torch.cuda.empty_cache()
    rep.table(rows, {"offered_req/s": "{:.0f}", "throughput_req/s": "{:.1f}", "out_tok/s": "{:.0f}",
                     "goodput_req/s": "{:.1f}", "slo_met": "{:.2f}", "ttft_p99_ms": "{:.0f}", "tpot_p99_ms": "{:.1f}"})
    rep("throughput counts every finished request; goodput only those that met both SLOs; slo_met is out of all")
    rep("offered requests (a shed request counts as a miss). Both are per second of wall time of that run.")
    rep.save({"rows": rows, "ttft_slo": args.ttft_slo, "tpot_slo": args.tpot_slo})


if __name__ == "__main__":
    main()
