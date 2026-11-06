"""Follow one batch through the model and print the shape and scale of every intermediate tensor.

    python c04-llm/scripts/inspect_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyllm import ByteTokenizer, TinyLLM, get_config  # noqa: E402
from tinyllm.attention_naive import split_heads  # noqa: E402
from tinyllm.rope import rotate  # noqa: E402


def show(name: str, t: torch.Tensor) -> None:
    print(f"{name:<34} {str(tuple(t.shape)):<20} rms={t.float().pow(2).mean().sqrt().item():.3f}")


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    config = get_config("tiny")
    model = TinyLLM(config).eval()
    tok = ByteTokenizer()
    ids = torch.tensor([tok.encode("The cat sat")])  # (1, 11)
    print(f"text 'The cat sat' -> ids {ids[0].tolist()}\n")

    with torch.no_grad():
        x = model.embed_tokens(ids)
        show("embeddings x", x)
        positions = torch.arange(ids.shape[1])[None]
        cos, sin = model.rotary(positions)
        show("rope cos / sin", cos)

        block = model.layers[0]
        attn = block.self_attn
        h = block.input_layernorm(x)
        show("block0 input_layernorm(x)", h)
        q = split_heads(attn.q_proj(h), attn.num_heads)
        k = split_heads(attn.k_proj(h), attn.num_kv_heads)
        v = split_heads(attn.v_proj(h), attn.num_kv_heads)
        show("q (B, H, T, d)", q)
        show("k (B, H_kv, T, d)", k)
        show("v (B, H_kv, T, d)", v)
        show("q after RoPE", rotate(q, cos, sin))
        attn_out = attn(h, cos, sin)
        show("attention output (after o_proj)", attn_out)
        x = x + attn_out
        show("residual after attention", x)
        g = block.mlp.gate_proj(block.post_attention_layernorm(x))
        show("mlp gate_proj (B, T, F)", g)
        x = x + block.mlp(block.post_attention_layernorm(x))
        show("residual after mlp = block0 out", x)

        out = model(ids, labels=ids, use_cache=True)
        show("logits (B, T, V)", out.logits)
        show("cache layer0 keys", out.past_key_values.keys[0])
        print(f"\nloss {out.loss.item():.4f}  (ln V = {torch.log(torch.tensor(256.0)).item():.4f})")
        print(f"cache holds {out.past_key_values.nbytes()} bytes for {ids.shape[1]} tokens")


if __name__ == "__main__":
    main()
