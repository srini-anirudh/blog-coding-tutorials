"""The KV cache became the constraint: the post's byte counts, decode intensity, and MLA's absorption trick.

Post sections: "The KV cache became the constraint", "The bandwidth argument", "Multi-head latent attention,
derived". The large-model numbers are arithmetic from published configs. The formula is checked against the bytes
that C4's real KVCache holds after a forward pass. The decode-time bandwidth measurement is in benchmarks/gpu_bench.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from organs.arith import (GIB, KIB, MIB, decode_intensity, kv_bytes, mha_equivalent_bytes_per_token,
                          mla_bytes_per_token, ridge_point)
from organs.cache import mla_absorption_check
from organs.report import Report
from tinyllm import ModelConfig, TinyLLM


def main() -> None:
    rep = Report("06_kv_cache")
    rep("1. bytes_KV = 2 L n_kv d_h s b bytes. Llama 3 8B shape: L = 32, d_h = 128, bf16.")
    rows = []
    for name, nkv in [("MHA, n_kv = 32", 32), ("GQA, n_kv = 8 (shipped)", 8), ("MQA, n_kv = 1", 1)]:
        per = kv_bytes(32, nkv, 128, 1)
        rows.append([name, f"{per / KIB:.0f} KiB", f"{kv_bytes(32, nkv, 128, 131072) / GIB:.1f} GiB",
                     f"{decode_intensity(32, nkv):.0f}"])
    rep.table(["attention", "per token", "128k tokens", "decode FLOPs/byte"], rows)
    rep(f"Weights: 8.03e9 params x 2 bytes = {8.03e9 * 2 / 1e9:.1f} GB ({8.03e9 * 2 / GIB:.1f} GiB). "
        f"H100 SXM ridge point: 989e12 / 3.35e12 = {ridge_point(989e12, 3.35e12):.0f} FLOPs/byte.")
    rep("Decode intensity does not depend on d_h or s:")
    rows = [[dh, s, f"{decode_intensity(32, 8, dh, s):.2f}"] for dh, s in [(64, 1024), (128, 8192), (256, 131072)]]
    rep.table(["d_h", "s", "FLOPs/byte for 32 q heads over 8 KV heads"], rows)
    rep("")

    rep("2. Check the formula against real cache tensors: C4's TinyLLM, 4 layers, 8 q heads, 2 KV heads, d_h 32,")
    rep("   one prefill of 96 tokens, batch 3, fp32 (4 bytes).")
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, hidden_size=256, num_layers=4, num_heads=8, num_kv_heads=2,
                      intermediate_size=512, max_seq_len=128)
    model = TinyLLM(cfg).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 256, (3, 96)), use_cache=True)
    held = out.past_key_values.nbytes()
    formula = kv_bytes(4, 2, 32, 96, batch=3, bytes_per=4)
    rep(f"   bytes held by the cache: {held:,}   formula: {formula:,}   equal: {held == formula}")
    rep("")

    rep("3. MLA, DeepSeek-V3 shape: 61 layers, r = 512, decoupled RoPE key of 64, 128 heads, K 192 + V 128, bf16.")
    mla = mla_bytes_per_token(61, 512, 64)
    mha = mha_equivalent_bytes_per_token(61, 128, 192, 128)
    rep(f"   MLA per token: {mla:,} B = {mla / KIB:.1f} KiB     equivalent MHA: {mha:,} B = {mha / MIB:.2f} MiB")
    rep(f"   ratio {mha / mla:.1f}x     at 128k tokens: {mla * 131072 / GIB:.1f} GiB vs {mha * 131072 / GIB:.0f} GiB")
    rep("")
    rep("4. Absorption, on one random token (d = 64, latent r = 16, d_h = 32, float64):")
    chk = mla_absorption_check()
    rep(f"   q^T (W_UK c) = {chk['materialized']:+.10f}")
    rep(f"   (W_UK^T q)^T c = {chk['absorbed']:+.10f}   (same number, K never built)")
    rep(f"   With RoPE on the key: true logit at key position 9          {chk['rope_materialized']:+.6f}")
    rep(f"   absorbed matrix built at position 5, used at position 9      {chk['rope_absorbed_wrong_pos']:+.6f}")
    rep(f"   ||R_9 W_UK - W_UK|| / ||W_UK|| = {chk['commutator_norm']:.3f}: the rotation does not pass through W_UK,")
    rep("   so an absorbed matrix would be needed per position. That is why DeepSeek adds a separate RoPE key.")
    rep("   C3 (c03-attention/attnzoo/mla.py) builds the full MLA layer with the decoupled key and absorb=True.")
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    main()
