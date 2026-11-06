"""The attention zoo on one laptop: training, prefill and decode for each architecture.

Run:  python c03-attention/benchmarks/bench_zoo.py [--device cpu mps] [--lengths 512 1024 ...]
One attention layer, D = 256, Hq = 8, dh = 32. Each cell runs in a fresh process.
"""

from __future__ import annotations

import argparse
import json

import _common
import torch

from attnzoo.config import AttentionConfig
from attnzoo.device import synchronize
from attnzoo.library import build_attention, make_cache

D, H = 256, 8
ZOO: dict[str, dict] = {
    "MHA": dict(attention_type="mha"),
    "GQA Hkv=2": dict(attention_type="gqa", num_kv_heads=2),
    "MQA": dict(attention_type="mqa"),
    "MLA r=64": dict(attention_type="mla", kv_lora_rank=64, qk_rope_head_dim=8),
    "local mask": dict(attention_type="gqa", num_kv_heads=2, window_size=256),
    "local chunked": dict(attention_type="gqa", num_kv_heads=2, window_size=256, backend="window_chunked"),
    "linear": dict(attention_type="linear"),
    "GQA naive": dict(attention_type="gqa", num_kv_heads=2, backend="naive"),
}
DECODE_STEPS = 16


def child(a: dict) -> None:
    torch.set_num_threads(2)
    device = torch.device(a["device"])
    n = a["n"]
    cfg = AttentionConfig(hidden_size=D, num_heads=H, max_seq_len=n + DECODE_STEPS + 1, **ZOO[a["name"]])
    model = build_attention(cfg).to(device)
    x = torch.randn(1, n, D, device=device)
    target = torch.randn(1, n, D, device=device)
    repeats = 3 if n <= 2048 else 1

    # Training: forward and backward, separately timed.
    fwd, bwd = [], []
    with _common.PeakMemory(device) as mem:
        for _ in range(repeats):
            synchronize(device)
            t0 = _common.time.perf_counter()
            loss = (model(x) - target).pow(2).mean()
            synchronize(device)
            t1 = _common.time.perf_counter()
            loss.backward()
            synchronize(device)
            t2 = _common.time.perf_counter()
            fwd.append((t1 - t0) * 1e3)
            bwd.append((t2 - t1) * 1e3)
            model.zero_grad(set_to_none=True)
    del loss

    # Prefill then decode. The linear layer's parallel form has no cache, so prefill it step by step
    # only for short lengths; its recurrence is what decoding uses.
    model.eval()
    with torch.inference_mode():
        cache = make_cache(cfg)
        synchronize(device)
        t0 = _common.time.perf_counter()
        model(x, kv_cache=cache)
        synchronize(device)
        prefill_ms = (_common.time.perf_counter() - t0) * 1e3
        step = torch.randn(1, 1, D, device=device)
        synchronize(device)
        t0 = _common.time.perf_counter()
        for _ in range(DECODE_STEPS):
            model(step, kv_cache=cache)
        synchronize(device)
        decode_ms = (_common.time.perf_counter() - t0) * 1e3 / DECODE_STEPS
    print(json.dumps({
        "fwd": _common.statistics.median(fwd), "bwd": _common.statistics.median(bwd), "peak": mem.bytes,
        "prefill_tok_s": n / (prefill_ms / 1e3), "prefill_ms": prefill_ms, "decode": decode_ms, "cache": cache.nbytes(),
        "params": sum(p.numel() for p in model.parameters()),
    }))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--child")
    p.add_argument("--device", nargs="+", default=["cpu", "mps"])
    p.add_argument("--lengths", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192])
    args = p.parse_args()
    if args.child:
        return child(json.loads(args.child))

    rows = []
    for device in args.device:
        print(f"\n{_common.hardware_note(device)}. One layer, D = {D}, Hq = {H}, dh = 32.")
        print(f"{'layer':15}{'N':>6}{'fwd ms':>10}{'bwd ms':>10}{'peak MB':>9}{'prefill ms':>12}{'tok/s':>10}{'decode ms/tok':>15}{'cache MB':>10}")
        for name in ZOO:
            for n in args.lengths:
                # The naive backend stores (N, N) per head with autograd: stop before it eats the machine.
                if name == "GQA naive" and n > 4096:
                    continue
                r = _common.run_child(__file__, {"device": device, "n": n, "name": name}, timeout=300)
                if "error" in r:
                    print(f"{name:15}{n:>6}  {r['error']}", flush=True)
                    continue
                print(f"{name:15}{n:>6}{r['fwd']:>10.1f}{r['bwd']:>10.1f}{r['peak'] / 2**20:>9.0f}{r['prefill_ms']:>12.1f}"
                      f"{r['prefill_tok_s']:>10,.0f}{r['decode']:>15.2f}{r['cache'] / 2**20:>10.2f}", flush=True)
                rows.append({"device": device, "name": name, "n": n, **r})
    from pathlib import Path

    out = Path(__file__).resolve().parent.parent / "results" / "bench_zoo.json"
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
