"""Sections "The bill" and "When thought stops being text".

1. Checks C_gen = 2NK + 4Ld(PK + K(K+1)/2) against torch's FLOP counter on a real TinyLLM decode loop,
   and the KV-cache formula against the bytes the cache actually holds.
2. Recomputes every number the post quotes for a 32B-class dense model from its config.
3. Prints the bill for growing thinking lengths.
4. Recomputes log2 V for Qwen3's tokenizer (from the Hub cache if present).

  python labs/thinking-in-tokens/scripts/02_cost_of_thought.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

import torch  # noqa: E402

from thinktok.cost import (  # noqa: E402
    DENSE_32B, H100_SXM_BANDWIDTH, bits_per_step, count_decode_flops, decode_floor_seconds, gen_flops,
    kv_bytes_per_token, matmul_params, non_embedding_params, per_token_attention_share,
)
from thinktok.util import hardware  # noqa: E402
from tinyllm import PRESETS, ModelConfig, TinyLLM  # noqa: E402

QWEN3_VOCAB_IN_POST = 151_669


def qwen3_vocab() -> tuple[int, str]:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
        return len(tok), "len(AutoTokenizer('Qwen/Qwen3-0.6B'))"
    except Exception:  # not cached: fall back to the number the post quotes
        return QWEN3_VOCAB_IN_POST, "value quoted in the post (tokenizer not in the local cache)"


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    out: list[str] = [f"# hardware: {hardware()}", "# command: python labs/thinking-in-tokens/scripts/02_cost_of_thought.py", ""]
    data: dict = {}

    # 1. Formula versus FLOP counter on the C4 "tiny" preset with reference attention.
    cfg = ModelConfig(**{**PRESETS["tiny"].to_dict(), "attn_backend": "naive", "max_seq_len": 2048})
    model = TinyLLM(cfg)
    out += ["## 1. Formula vs torch FLOP counter (TinyLLM 'tiny' preset, fp32, batch 1, cached decode)", "",
            f"matmul weights per token N = {matmul_params(cfg):,} (attention + MLP + LM head)", "",
            "| prompt P | thinking K | counted FLOPs | 2NK + 4Ld(PK+K(K+1)/2) | post's K^2/2 form | attention share | KV bytes held | KV formula |",
            "|---|---|---|---|---|---|---|---|"]
    rows = []
    for p, k in [(16, 16), (64, 256), (256, 256), (16, 1024)]:
        r = count_decode_flops(model, p, k)
        rows.append(r)
        out.append(f"| {p} | {k} | {r['measured_flops']:,} | {r['formula_exact']:,.0f} | {r['formula_post']:,.0f} | "
                   f"{r['attention_term'] / r['formula_exact']:.1%} | {r['kv_bytes_measured']:,} | {r['kv_bytes_formula']:,} |")
    data["tinyllm_check"] = rows

    # 2. The post's quoted numbers, recomputed.
    n32 = non_embedding_params(DENSE_32B)
    kv_tok = kv_bytes_per_token(64, 8, 128, 2)
    claims = [
        ("non-embedding params N of the 32B-class config", "32B-class", f"{n32 / 1e9:.1f} B"),
        ("attention / weight FLOPs per token at 1k context", "a few per cent", f"{per_token_attention_share(DENSE_32B, 1024):.1%}"),
        ("attention / weight FLOPs per token at 32k context", "roughly comparable", f"{per_token_attention_share(DENSE_32B, 32768):.0%}"),
        ("context where the two terms are equal", "-", f"{n32 / (2 * 64 * 5120):,.0f} tokens"),
        ("KV cache per token, 64 layers x 8 KV heads x 128, bf16", "256 KiB", f"{kv_tok / 1024:.0f} KiB"),
        ("KV cache for 10,000 thinking tokens", "about 2.6 GB", f"{kv_tok * 10_000 / 1e9:.2f} GB"),
        ("weights in bf16, 32e9 params", "64 GB", f"{32e9 * 2 / 1e9:.0f} GB"),
        ("decode floor per token at 3.3 TB/s", "about 19 ms", f"{decode_floor_seconds(32e9, 2, 3.3e12) * 1e3:.1f} ms"),
        ("decode floor per token at 3.35 TB/s (H100 SXM spec)", "-", f"{decode_floor_seconds(32e9) * 1e3:.1f} ms"),
        ("10,000 thinking tokens at that floor", "three minutes", f"{decode_floor_seconds(32e9, 2, 3.3e12) * 1e4 / 60:.1f} min"),
    ]
    out += ["", "## 2. The post's numbers for a 32B-class dense model (arithmetic from the config, not a measurement)", "",
            "| quantity | post says | recomputed |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in claims]
    data["post_claims"] = claims

    # 3. The bill as thinking grows (prompt of 1,000 tokens).
    out += ["", "## 3. Bill for K thinking tokens, 32B-class model, P = 1,000, batch 1 (arithmetic)", "",
            "| K | weight TFLOPs | attention TFLOPs | attention share | KV cache GB | decode floor (s) |", "|---|---|---|---|---|---|"]
    bill = []
    for k in [1_000, 4_000, 10_000, 32_000]:
        b = gen_flops(n32, 64, 5120, 1_000, k)
        kv = kv_tok * (1_000 + k) / 1e9
        sec = decode_floor_seconds(32e9, 2, H100_SXM_BANDWIDTH) * k
        bill.append({"k": k, "weights": b.weights, "attention": b.attention, "kv_gb": kv, "seconds": sec})
        out.append(f"| {k:,} | {b.weights / 1e12:,.0f} | {b.attention / 1e12:,.0f} | {b.attention / b.total:.0%} | {kv:.1f} | {sec:.0f} |")
    out.append("")
    out.append("The decode floor ignores KV-cache reads, which also grow with K, so real batch-1 latency is higher.")
    data["bill"] = bill

    # 4. The token channel.
    v, src = qwen3_vocab()
    out += ["", "## 4. The token channel between reasoning steps", "",
            f"vocabulary V = {v:,} ({src})", f"log2 V = {bits_per_step(v):.2f} bits per step (post: about 17.2)"]
    data["bits"] = {"vocab": v, "source": src, "bits": bits_per_step(v)}

    text = "\n".join(out) + "\n"
    print(text)
    (LAB / "results" / "02_cost_of_thought.md").write_text(text)
    (LAB / "results" / "02_cost_of_thought.json").write_text(json.dumps(data, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
