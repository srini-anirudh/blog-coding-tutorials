"""The mask grew holes: a sliding-window cache that stops growing, and the receptive field through depth.

Post sections: "The mask grew holes" and "What it buys, in bytes".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from organs.arith import GIB, swa_bytes
from organs.cache import decode_with_window, full_window_reference, interleave, receptive_field
from organs.report import Report


def main() -> None:
    rep = Report("07_sliding_window")
    rep("1. A window-w layer only ever needs the last w keys. Decode 512 tokens one at a time through a")
    rep("   ring buffer of w slots, and compare with full attention under a sliding-window mask (d = 64, fp64).")
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(512, 64, generator=g, dtype=torch.float64) for _ in range(3))
    rows = []
    for w in (16, 64, 256):
        out, nbytes = decode_with_window(q, k, v, w)
        ref = full_window_reference(q, k, v, w)
        rows.append([w, f"{(out - ref).abs().max():.1e}", f"{nbytes:,}", f"{2 * 512 * 64 * 8:,}"])
    rep.table(["window", "max |ring - masked|", "ring cache bytes", "full cache bytes at t=512"], rows)
    rep("")

    rep("2. Olmo 3 32B at 65,536 tokens: 64 layers, 8 KV heads, d_h 128, bf16 -> 4096 bytes per layer per token.")
    per = 2 * 8 * 128 * 2
    full = swa_bytes(64, 0, 4096, 65536, per)
    mixed = swa_bytes(16, 48, 4096, 65536, per)
    rep(f"   all global: {full / GIB:.2f} GiB   3:1 with window 4096: {mixed / GIB:.2f} GiB   ratio {full / mixed:.2f}x")
    rows = []
    for s in (4096, 16384, 65536, 262144):
        rows.append([f"{s:,}", f"{swa_bytes(64, 0, 4096, s, per) / GIB:.2f}", f"{swa_bytes(16, 48, 4096, s, per) / GIB:.2f}"])
    rep.table(["context", "all global GiB", "3:1 SWA GiB"], rows)
    rep("   Past the window, only the 16 global layers keep growing.")
    rep("")

    rep("3. Receptive field: how far back the last of 1024 tokens can see after l layers, window 32.")
    rep("   Computed by composing the boolean masks layer by layer (a path through the masks).")
    schemes = [("all local (Mistral)", None), ("1:1 (gpt-oss)", 1), ("3:1 (Olmo 3)", 3), ("5:1 (Gemma 3)", 5)]
    layers = [1, 2, 4, 6, 8, 12]
    rows = []
    for name, ratio in schemes:
        reach = receptive_field(interleave(12, ratio), 1024, 32)
        rows.append([name] + [reach[l - 1] for l in layers])
    rep.table(["scheme"] + [f"l={l}" for l in layers], rows)
    rep("   All-local grows by w - 1 = 31 tokens per layer: L (w - 1) after L layers. One global layer reaches")
    rep("   everything (1023 = the whole prefix).")
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    main()
