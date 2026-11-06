"""Patch n' Pack and the missing pad mask, on a tiny 2D-RoPE ViT with random weights.

Writes results/05_patch_n_pack.txt and results/05_patch_n_pack.json.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vencoder.common import table, write_json, write_text  # noqa: E402
from vencoder.intake import plan_qwen  # noqa: E402
from vencoder.packing import TinyViT, mean_pool, pack, pad_batch, packing_cost, rope_2d  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    lines, out = [], {}
    model = TinyViT().eval()
    shapes = [(32, 48), (64, 32), (40, 40), (24, 72)]  # H x W, patch 8
    images = [torch.rand(3, h, w) for h, w in shapes]

    with torch.no_grad():
        alone = [model.encode_one(img) for img in images]
        p = pack(model, images)
        packed = model(p["x"], p["rows"], p["cols"], p["mask"])[0]
        unmasked = model(p["x"], p["rows"], p["cols"], None)[0]
    rows = []
    for i, ref in enumerate(alone):
        sel = p["segment"] == i
        a, b = packed[sel], unmasked[sel]
        rows.append([f"{shapes[i][0]}x{shapes[i][1]}", len(ref), f"{(a - ref).abs().max():.1e}",
                     f"{(b - ref).abs().max():.2f}",
                     f"{F.cosine_similarity(b.mean(0), ref.mean(0), dim=0):.3f}"])
    out["pack"] = rows
    lines += [f"Four images packed into one sequence of {p['x'].shape[1]} patches, against each image encoded alone:"]
    lines += table(rows, ["image", "patches", "max diff, block-diag mask", "max diff, no mask", "pooled cosine, no mask"])

    # A 32 x 32 image (16 patches) padded with zero patches. With the mask nothing changes. Without it, it rots.
    img = torch.rand(3, 32, 32)
    with torch.no_grad():
        ref = model.encode_one(img)
        rows, pad_rows = [], {}
        for extra in (0, 16, 48, 112, 240):
            b = pad_batch(model, [img] + ([torch.zeros(3, 8, 8 * (16 + extra))] if extra else []))
            masked = model(b["x"], b["rows"], b["cols"], b["mask"])[0]
            nomask = model(b["x"], b["rows"], b["cols"], None)[0]
            v = b["valid"][:1]
            good = mean_pool(masked[None], v)[0]
            bad = mean_pool(nomask[None], None)[0]
            r = [16 + extra, f"{extra / (16 + extra):.0%}", f"{(masked[:16] - ref).abs().max():.1e}",
                 f"{F.cosine_similarity(good, ref.mean(0), dim=0):.4f}",
                 f"{F.cosine_similarity(nomask[:16].mean(0), ref.mean(0), dim=0):.4f}",
                 f"{F.cosine_similarity(bad, ref.mean(0), dim=0):.4f}"]
            rows.append(r)
    out["pad"] = rows
    lines += ["", "One 16-patch image in a batch padded to the longest row (padding = zero-pixel patches):"]
    lines += table(rows, ["row length", "padding", "max diff, masked", "pooled cos, masked",
                          "pooled cos, no mask, real only", "pooled cos, no mask, all"])

    # Compute waste on a realistic size mix at native resolution.
    rnd = random.Random(0)
    aspects = [1 / 1.41, 4 / 3, 16 / 9, 1 / 3.2, 3.0, 1.0]
    lengths = []
    for _ in range(512):
        long = rnd.uniform(400, 1600)
        a = rnd.choice(aspects)
        w, h = (long, long / a) if a >= 1 else (long * a, long)
        lengths.append(plan_qwen(int(w), int(h)).patches)
    cost = packing_cost(lengths, batch=8, capacity=max(lengths))
    out["packing_cost"] = cost
    lines += ["", f"512 images, long side uniform in 400-1600 px, six aspect ratios, Qwen2-VL patch counts "
                  f"({min(lengths)} to {max(lengths)} patches):",
              f"  pad to longest, batches of 8: {cost['padded_slots']:,} slots, {cost['padded_waste']:.0%} padding",
              f"  first-fit packing into {cost['packed_sequences']} sequences of {max(lengths):,}: "
              f"{cost['packed_slots']:,} slots, {cost['packed_waste']:.0%} padding",
              f"  attention query-key pairs: dense padded {cost['attn_pairs_padded']:.3e}, "
              f"block-diagonal {cost['attn_pairs_block_diag']:.3e} "
              f"({cost['attn_pairs_padded'] / cost['attn_pairs_block_diag']:.1f}x fewer)"]

    # 2D RoPE only sees relative offsets: shifting both positions leaves q.k unchanged.
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    dots = []
    for shift in [(0, 0), (5, 3), (40, 17)]:
        qr = rope_2d(q, torch.tensor([[2 + shift[0]]]), torch.tensor([[1 + shift[1]]]))
        kr = rope_2d(k, torch.tensor([[7 + shift[0]]]), torch.tensor([[4 + shift[1]]]))
        dots.append(float((qr * kr).sum()))
    out["rope_shift_dots"] = dots
    lines += ["", "2D RoPE: q.k for the same (5, 3) offset at three absolute positions: "
              + ", ".join(f"{d:.5f}" for d in dots)]

    print("\n".join(lines))
    write_text("05_patch_n_pack.txt", lines)
    write_json("05_patch_n_pack.json", out)


if __name__ == "__main__":
    main()
