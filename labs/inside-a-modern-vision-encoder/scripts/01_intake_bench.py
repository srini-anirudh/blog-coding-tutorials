"""The intake bench, the token cost calculator and the 1200 x 1600 walkthrough page, computed for real.

Writes results/01_intake_bench.txt and results/01_intake_bench.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vencoder.common import table, write_json, write_text  # noqa: E402
from vencoder.intake import (  # noqa: E402
    QWEN_CODE_DEFAULT_MAX_PIXELS, SHAPES, STRATEGIES, bench, plan_clip_crop, plan_qwen, vit_flops,
)

# Token cost calculator presets. The post does not give the receipt's size; 600 x 3000 is our choice.
PRESETS = {
    "1 MP square": (1024, 1024),
    "A4 @ 200 dpi": (1654, 2339),
    "1440p screenshot": (2560, 1440),
    "4K frame": (3840, 2160),
    "Long receipt": (600, 3000),
}
# Qwen2.5-VL vision tower, from the Qwen2.5-VL-3B-Instruct config.json on the Hub.
QWEN25_VIT = dict(dim=1280, mlp=3420, depth=32, window=(112 // 14) ** 2, full_layers=4)


def main() -> None:
    lines, out = [], {}

    lines += ["Intake bench: 1024 px long side, bar groups at 24/15/9/5/3 px.",
              "resolved = bar groups with >= 50% of the black-white swing left at the worst of 4 pixel phases;",
              "proxy = the post's rule, stroke >= 1.6 output px. Both count only groups that were not cropped.", ""]
    rows, cells = [], []
    for shape in SHAPES:
        for key in STRATEGIES:
            r = bench(shape, key)
            cells.append(r)
            rows.append([shape, r["strategy"], r["patches"], r["llm_tokens"], r["views"],
                         f"{r['scale_x']:.3f} x {r['scale_y']:.3f}", r["distortion_pct"], r["discarded_pct"],
                         r["padding_pct"], f"{r['resolved_measured']}/5", f"{r['resolved_proxy']}/5"])
    lines += table(rows, ["image", "strategy", "patches", "LLM tokens", "views", "sampled at (x, y)",
                          "stretch %", "cropped %", "padding %", "resolved", "proxy"])
    out["bench"] = cells

    lines += ["", "Token cost calculator (patches -> LLM tokens). Fixed-resolution rows ignore the size."]
    rows, calc = [], {}
    for name, (w, h) in PRESETS.items():
        row = [f"{name} {w}x{h}"]
        for key, fn in STRATEGIES.items():
            p = fn(w, h)
            calc.setdefault(name, {})[key] = {"patches": p.patches, "llm_tokens": p.llm_tokens}
            row.append(f"{p.patches}->{p.llm_tokens}" if p.patches != p.llm_tokens else str(p.patches))
        rows.append(row)
    lines += table(rows, ["image", *[STRATEGIES[k](1, 1).name for k in STRATEGIES]])
    out["calculator"] = calc

    # The walkthrough page: a photographed A4 page, 1200 x 1600.
    w, h = 1200, 1600
    dino = plan_clip_crop(w, h, 224, 16)
    qwen = plan_qwen(w, h)
    qwen_capped = plan_qwen(w, h, max_pixels=QWEN_CODE_DEFAULT_MAX_PIXELS)
    lines += ["", f"Walkthrough page {w}x{h}:",
              f"  DINO v1 ViT-B/16 @224 centre crop: {dino.patches} patches",
              f"  Qwen2-VL, Hub config max_pixels=12845056: {qwen.resize_to[0]}x{qwen.resize_to[1]} -> "
              f"{qwen.patches} patches -> {qwen.llm_tokens} LLM tokens after the 2x2 merge",
              f"  spread: {qwen.patches / dino.patches:.1f}x",
              f"  Qwen2-VL with the transformers code default max_pixels={QWEN_CODE_DEFAULT_MAX_PIXELS}: "
              f"{qwen_capped.resize_to[0]}x{qwen_capped.resize_to[1]} -> {qwen_capped.patches} patches "
              f"({qwen_capped.patches / qwen.patches:.0%} of the uncapped count)"]
    out["walkthrough"] = {"dino_v1_patches": dino.patches, "qwen_patches": qwen.patches,
                          "qwen_llm_tokens": qwen.llm_tokens, "qwen_capped_patches": qwen_capped.patches}

    lines += ["", f"Patch size: p=14 gives {(16 / 14) ** 2 - 1:.1%} more tokens than p=16 at equal pixels."]

    # Qwen2.5-VL window attention: cost of the ViT on the walkthrough page and on a 4K frame.
    lines += ["", "Qwen2.5-VL ViT forward FLOPs (32 layers, d=1280; window = 8x8 patches, full attention at 4 layers):"]
    rows, flops = [], {}
    for name, (fw, fh) in {"page 1200x1600": (w, h), "4K frame 3840x2160": (3840, 2160)}.items():
        n = plan_qwen(fw, fh).patches
        full = vit_flops(n, QWEN25_VIT["dim"], QWEN25_VIT["mlp"], QWEN25_VIT["depth"])
        win = vit_flops(n, **QWEN25_VIT)
        flops[name] = {"patches": n, "all_full": full, "windowed": win}
        rows.append([name, n, f"{full['total'] / 1e12:.1f}", f"{full['attention'] / full['total']:.0%}",
                     f"{win['total'] / 1e12:.1f}", f"{win['attention'] / win['total']:.0%}",
                     f"{full['total'] / win['total']:.2f}x"])
    lines += table(rows, ["image", "patches", "all-full TFLOPs", "attn share", "windowed TFLOPs", "attn share",
                          "saving"])
    out["qwen25_vit_flops"] = flops

    print("\n".join(lines))
    write_text("01_intake_bench.txt", lines)
    write_json("01_intake_bench.json", out)


if __name__ == "__main__":
    main()
