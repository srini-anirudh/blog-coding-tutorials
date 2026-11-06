"""Measure the five stages of one image request on Qwen2-VL-2B, and set them next to the cost model.

Token counts are checked against `costmodel`. FLOPs come from the same formulas with Qwen2-VL-2B's sizes
(675M encoder with full attention in all 32 layers; 1.5B language model, 28 layers, 2 KV heads x 128).
"""

import argparse

import torch
from _common import quiet_transformers, write

from pixelcost import costmodel as cm
from pixelcost import qwen
from pixelcost.synth import QUESTIONS, make_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sizes", default="448x448,672x672,1008x1344,1344x1792")
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--eager", action="store_true", help="also time prefill with the eager (materialised) attention")
    args = ap.parse_args()
    quiet_transformers()
    torch.manual_seed(0)
    model, processor = qwen.load(device=args.device)

    vis = sum(p.numel() for p in model.model.visual.parameters())
    lm = sum(p.numel() for p in model.model.language_model.parameters())
    emb = model.model.language_model.embed_tokens.weight.numel()
    tc, vc = model.config.text_config, model.config.vision_config
    enc = cm.Encoder(vis, vc.depth, vc.embed_dim, global_layers=vc.depth)
    llm = cm.LanguageModel(lm - emb, tc.num_hidden_layers, tc.hidden_size, tc.num_key_value_heads,
                           tc.hidden_size // tc.num_attention_heads,
                           bytes_per_number=2 if args.device.startswith("cuda") else 4)

    rows, lines = [], [f"Qwen2-VL-2B: encoder {vis / 1e6:.0f}M params, language model {(lm - emb) / 1e9:.2f}B "
                       f"non-embedding params, device {args.device}, "
                       f"{'BF16' if args.device.startswith('cuda') else 'FP32'}"]
    head = (f"{'image':>10} {'patches':>7} {'tokens':>6} {'model':>6} {'prep ms':>8} {'enc ms':>8} {'pref ms':>8} "
            f"{'dec ms/tok':>10} {'enc TF/s':>8} {'pref TF/s':>9} {'KV MB':>7} {'model MB':>8}")
    lines.append(head)
    for size in args.sizes.split(","):
        h, w = map(int, size.split("x"))
        img = make_sample(0, side=max(h, w)).image.resize((w, h))
        t = qwen.time_stages(model, processor, img, QUESTIONS["color"], args.decode_steps, args.repeats)
        pred_tokens = cm.visual_tokens(h, w)
        enc_flops = enc.flops(t["patches"])
        pre_flops = llm.prefill_flops(t["prompt_tokens"])
        row = dict(image=f"{w}x{h}", **t, model_visual_tokens=pred_tokens, encoder_tflop=enc_flops / 1e12,
                   prefill_tflop=pre_flops / 1e12, encoder_tflops_per_s=enc_flops / t["encode_s"] / 1e12,
                   prefill_tflops_per_s=pre_flops / t["prefill_s"] / 1e12,
                   model_kv_bytes=llm.kv_bytes_per_token() * t["prompt_tokens"])
        if args.eager:
            inputs = qwen.build_inputs(processor, img, QUESTIONS["color"], args.device)
            p = qwen.prepare(model, inputs, qwen.encode(model, inputs))
            row["prefill_sdpa_s"] = t["prefill_s"]
            row["prefill_eager_s"] = time_prefill_eager(model, p, args.repeats, args.device)
        rows.append(row)
        lines.append(f"{row['image']:>10} {t['patches']:>7} {t['visual_tokens']:>6} {pred_tokens:>6} "
                     f"{t['preprocess_s'] * 1e3:>8.1f} {t['encode_s'] * 1e3:>8.1f} {t['prefill_s'] * 1e3:>8.1f} "
                     f"{t['decode_step_s'] * 1e3:>10.2f} {row['encoder_tflops_per_s']:>8.2f} "
                     f"{row['prefill_tflops_per_s']:>9.2f} {t['kv_bytes'] / 1e6:>7.1f} "
                     f"{row['model_kv_bytes'] / 1e6:>8.1f}")

    lines.append("\nMeasured latency for one request, split by stage, for answers of 1, 10 and 200 tokens")
    lines.append(f"{'image':>10} {'answer':>6} {'total ms':>9} {'prep+enc+prefill':>17} {'decode':>7}")
    for r in rows:
        front = r["preprocess_s"] + r["encode_s"] + r["prefill_s"]
        for n in (1, 10, 200):
            total = front + n * r["decode_step_s"]
            lines.append(f"{r['image']:>10} {n:>6} {total * 1e3:>9.0f} {front / total:>17.0%} "
                         f"{n * r['decode_step_s'] / total:>7.0%}")
    if args.eager:
        lines.append("\nSame prefill FLOPs, two attention kernels (eager writes the attention matrix out)")
        for r in rows:
            lines.append(f"{r['image']:>10} sdpa {r['prefill_sdpa_s'] * 1e3:8.1f} ms   "
                         f"eager {r['prefill_eager_s'] * 1e3:8.1f} ms   ratio {r['prefill_eager_s'] / r['prefill_sdpa_s']:.2f}")
    lines.append("\nTimes are medians of the repeats after one warm-up. 'model' columns come from costmodel.py.")
    tag = "cuda" if args.device.startswith("cuda") else "cpu"
    write(f"03_stage_timing_{tag}", "\n".join(lines), {"rows": rows}, device=args.device)


def time_prefill_eager(model, p, repeats, device):
    import time
    cfg = model.model.language_model.config
    old = cfg._attn_implementation
    cfg._attn_implementation = "eager"
    try:
        times = []
        for _ in range(repeats + 1):
            qwen.sync(device); t0 = time.perf_counter()
            qwen.prefill(model, p.embeds, p.positions)
            qwen.sync(device); times.append(time.perf_counter() - t0)
    finally:
        cfg._attn_implementation = old
    times = sorted(times[1:])
    return times[len(times) // 2]


if __name__ == "__main__":
    main()
