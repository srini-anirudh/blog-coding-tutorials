"""Post sections 3, 10 and 7: prefill against decode, the decode wall, and what full MHA costs.

Builds a Llama-3-8B-shaped decoder with random bf16 weights (the speed of a step does not depend on the weight
values) and measures:

1. prefill of 2,048 and 8,192 prompt tokens: time, achieved TFLOP/s, and the post's 335 ms estimate;
2. one decode step at batch 1 as the context grows, eager and as a CUDA graph, against the floor Q / beta;
3. the same decode step for a Llama-2-7B-shaped model (full MHA, 4x the cache), including the 128k case
   that the post says does not fit on one 80 GB card.

    python labs/the-mechanics-of-llm-inference/scripts/04_decode_wall.py --out .../results/decode_wall_h100.txt
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferlab.decoder import SHAPES, Decoder  # noqa: E402
from inferlab.ledger import ACCELERATORS  # noqa: E402
from inferlab.timing import Report, cuda_graph, measured_bandwidth, time_ms  # noqa: E402


def kv_bytes(model: Decoder, batch: int, positions: int) -> int:
    c = model.config
    return 2 * c.layers * c.n_kv * c.head_dim * 2 * batch * positions


def free(model: Decoder) -> None:
    model.k_cache = model.v_cache = None
    gc.collect()
    torch.cuda.empty_cache()


def decode_sweep(rep: Report, model: Decoder, contexts: list[int], bw: float, label: str) -> None:
    spec = ACCELERATORS["h100"]
    w = model.weight_bytes(read_per_decode_step=True)
    rep(f"\n{label}: decode step at batch 1. Weights read per step {w / 1e9:.2f} GB (embedding table excluded).")
    rep(f"{'context':>8} {'KV GB':>6} {'floor spec':>10} {'floor meas':>10} {'eager ms':>9} {'graph ms':>9} "
        f"{'tok/s':>6} {'achieved':>9}")
    tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    for n in contexts:
        need = kv_bytes(model, 1, n + 1)
        try:
            free(model)
            model.allocate_cache(1, n + 1)
        except torch.cuda.OutOfMemoryError:
            free(model)
            rep(f"{n:>8} {need / 1e9:>6.2f}  does not fit: weights {model.weight_bytes() / 1e9:.1f} GB + cache "
                f"{need / 1e9:.1f} GB > {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
            continue
        eager = time_ms(lambda: model(tok, n), warmup=3, iters=10)
        replay = cuda_graph(lambda: model(tok, n))
        graph = time_ms(replay, warmup=3, iters=30)
        q = w + need
        rep(f"{n:>8} {need / 1e9:>6.2f} {q / spec.bandwidth * 1e3:>8.2f}ms {q / bw * 1e3:>8.2f}ms {eager:>9.2f} "
            f"{graph:>9.2f} {1e3 / graph:>6.0f} {q / (graph / 1e3) / 1e12:>6.2f}TB/s")
        del replay
    free(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="llama-3-8b")
    ap.add_argument("--mha-shape", default="llama-2-7b")
    ap.add_argument("--prefill", default="2048,8192")
    ap.add_argument("--contexts", default="512,2048,8192,32768,131072")
    ap.add_argument("--mha-contexts", default="8192,32768,131072")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")
    rep = Report(args.out)
    spec = ACCELERATORS["h100"]
    bw = measured_bandwidth()
    rep(f"measured copy bandwidth {bw / 1e12:.2f} TB/s (spec {spec.bandwidth / 1e12:.2f}). Random bf16 weights.")
    rep("floor spec = bytes / 3.35 TB/s, floor meas = bytes / measured copy bandwidth. graph = CUDA-graph replay.")

    model = Decoder.random(SHAPES[args.shape], max_len=8193, device="cuda", dtype=torch.bfloat16)
    params = sum(p.numel() for p in model.parameters())
    c = model.config
    rep(f"\n{args.shape}: {params / 1e9:.2f}B parameters, {model.weight_bytes() / 1e9:.2f} GB in bf16")

    rep("\nPrefill at batch 1 (LM head on the last position only)")
    rep(f"{'prompt':>7} {'ms':>8} {'TFLOP':>7} {'TFLOP/s':>8} {'% spec peak':>11} {'us/token':>9}")
    for n in [int(x) for x in args.prefill.split(",")]:
        free(model)
        model.allocate_cache(1, n)
        ids = torch.randint(0, c.vocab, (1, n), device="cuda")
        ms = time_ms(lambda: model(ids, 0), warmup=2, iters=5)
        flops = 2 * params * n + 2 * c.layers * n**2 * c.d_model  # the post's W_prefill
        rep(f"{n:>7} {ms:>8.1f} {flops / 1e12:>7.1f} {flops / (ms / 1e3) / 1e12:>8.0f} "
            f"{100 * flops / (ms / 1e3) / spec.peak_flops:>10.0f}% {ms * 1e3 / n:>9.1f}")
    rep("post's estimate for 8,192 tokens: 149 TFLOP at 45% of peak = 335 ms, 41 us per token")

    decode_sweep(rep, model, [int(x) for x in args.contexts.split(",")], bw, args.shape)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = Decoder.random(SHAPES[args.mha_shape], max_len=8193, device="cuda", dtype=torch.bfloat16)
    decode_sweep(rep, model, [int(x) for x in args.mha_contexts.split(",")], bw, args.mha_shape + " (full MHA)")
    rep.save()


if __name__ == "__main__":
    main()
