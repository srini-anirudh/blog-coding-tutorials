"""Count parameters by component from the formula, then verify against a real nn.Module.

Big models are built on the meta device: shapes only, no memory, so Llama-3-8B fits on a laptop.

    python c04-llm/scripts/param_count.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyllm import TinyLLM, count_parameters, get_config, kv_cache_bytes  # noqa: E402


def fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3f}B"
    return f"{n / 1e6:.2f}M" if n >= 1e5 else f"{n:,}"


def main() -> None:
    names = ["tiny", "smollm2-135m", "llama3-8b"]
    print(f"{'component':<12}" + "".join(f"{n:>16}" for n in names))
    rows = {n: count_parameters(get_config(n)) for n in names}
    for part in ["embeddings", "attention", "mlp", "norms", "lm_head", "total"]:
        print(f"{part:<12}" + "".join(f"{fmt(rows[n][part]):>16}" for n in names))

    print("\nverify: formula == sum(p.numel()) of the built module")
    for n in names:
        with torch.device("meta"):
            model = TinyLLM(get_config(n))
        actual = model.num_parameters()
        print(f"  {n:<14} formula={rows[n]['total']:>14,}  module={actual:>14,}  equal={rows[n]['total'] == actual}")

    print("\nKV cache per sequence of 4096 tokens, bf16 (2 bytes per value)")
    for label, kv in [("MHA (32 kv heads)", 32), ("GQA (8 kv heads)", 8), ("MQA (1 kv head)", 1)]:
        c = get_config("llama3-8b", num_kv_heads=kv)
        attn = count_parameters(c)["attention"]
        print(f"  {label:<20} cache={kv_cache_bytes(c, 4096) / 2**30:5.2f} GiB  attention params={fmt(attn)}")


if __name__ == "__main__":
    main()
