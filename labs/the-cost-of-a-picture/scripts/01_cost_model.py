"""Recompute the post's worked example and token-count rules from the formulas. No model, no measurement."""

from _common import write

from pixelcost import costmodel as cm


def main():
    lines, data = [], {}
    c = cm.request_cost(1008, 1344)            # the 1344 x 1008 photo, 40-token question
    kv_mha = cm.LanguageModel(7e9, 28, 3584, 28, 128).kv_bytes_per_token() * c.prompt
    rows = [
        ("patches / visual tokens", f"{c.patches:,} / {c.visual:,}", "6,912 / 1,728"),
        ("encoder compute", f"{c.encoder_flops / 1e12:.1f} TFLOP, {c.encoder_s * 1e3:.0f} ms", "9.3 TFLOP, 19 ms"),
        ("prefill compute (1,768 tokens)", f"{c.prefill_flops / 1e12:.1f} TFLOP, {c.prefill_s * 1e3:.0f} ms",
         "26 TFLOP, 53 ms"),
        ("KV cache per token", f"{cm.POST_LLM.kv_bytes_per_token() / 1024:.0f} KiB", "56 KiB"),
        ("KV cache, this request", f"{c.kv_bytes / 1e6:.0f} MB, image {c.image_kv_share:.0%}", "101 MB, 98%"),
        ("same model, 28 KV heads (no GQA)", f"{kv_mha / 1e6:.0f} MB", "710 MB"),
        ("one decode step, batch 1", f"{c.decode_step_s * 1e3:.2f} ms", "4.2 ms"),
        ("200-token answer, decode only", f"{200 * c.decode_step_s:.2f} s", "about 0.85 s"),
    ]
    lines.append("Worked example (600M encoder, 4 global layers; 7B LLM, 28 layers, 4 KV heads x 128; H100 at 50%)")
    lines.append(f"{'quantity':<36} {'this code':<28} post")
    lines += [f"{a:<36} {b:<28} {p}" for a, b, p in rows]
    data["worked_example"] = {a: b for a, b, _ in rows}

    lines.append("\nWhich stage dominates: estimated latency split for answers of 1, 10 and 200 tokens")
    lines.append(f"{'answer':>7} {'encoder':>9} {'prefill':>9} {'decode':>9}  {'image share of time':>20}")
    split = {}
    for n in (1, 10, 200):
        total = c.total_s(n)
        share = (c.encoder_s + c.prefill_s * c.image_kv_share) / total
        lines.append(f"{n:>7} {c.encoder_s * 1e3:>7.0f}ms {c.prefill_s * 1e3:>7.0f}ms "
                     f"{n * c.decode_step_s * 1e3:>7.0f}ms  {share:>19.0%}")
        split[n] = {"total_ms": round(total * 1e3, 1), "image_share": round(share, 3)}
    data["latency_split"] = split

    lines.append("\nConcurrency: how many such requests (200-token answers) fit in 80 GB next to the 7B weights")
    conc = {}
    for keep in (1.0, 0.5, 0.25, 0.111):
        ck = cm.request_cost(1008, 1344, keep=keep)
        n = cm.concurrent_requests(ck)
        conc[keep] = n
        lines.append(f"  keep {keep:>5.0%} of visual tokens ({ck.visual:>5} tokens): {n:>4} requests")
    data["concurrency"] = conc

    lines.append("\nTokens per image: connector merge (14-px patches), and three provider rules as the post states them")
    lines.append(f"{'image':>11} {'patches':>8} {'merge 1':>8} {'2x2':>6} {'3x3':>6} {'4x4':>6} "
                 f"{'28px rule':>10} {'512 tiles':>10} {'768 tiles':>10}")
    sizes = [(336, 336), (448, 448), (672, 672), (1000, 1000), (1008, 1344), (1080, 1920), (3508, 2480)]
    table = []
    for h, w in sizes:
        row = [cm.num_patches(h, w)] + [cm.visual_tokens(h, w, 14, m) for m in (1, 2, 3, 4)]
        row += [cm.anthropic_tokens(h, w), cm.openai_tokens(h, w), cm.gemini_tokens(h, w)]
        table.append({"h": h, "w": w, "values": row})
        lines.append(f"{f'{w}x{h}':>11} " + " ".join(f"{v:>{n},}" for v, n in zip(row, (8, 8, 6, 6, 6, 10, 10, 10))))
    data["token_counts"] = table
    lines.append("\nThe 3x3 and 4x4 merges floor the patch count; real connectors pad the grid first.")
    write("01_cost_model", "\n".join(lines), data)


if __name__ == "__main__":
    main()
