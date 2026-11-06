"""Preprocessing gotchas, measured: antialiasing, library disagreement, normalisation constants, EXIF, alpha.

Writes results/02_preprocessing.txt and results/02_preprocessing.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vencoder.common import table, write_json, write_text  # noqa: E402
from vencoder.preprocess import (  # noqa: E402
    NORMS, aliasing_report, exif_rotated_jpeg, flatten, library_disagreement, line_contrast, load, norm_mismatch,
    smooth_photo, stroke_page, transparent_chart, zone_plate,
)


def main() -> None:
    lines, out = [], {}

    # 1. A 4000 px page of 3 px strokes, shrunk to 336. The strokes are far below the output's Nyquist limit,
    #    so the right answer is a flat grey. Anything else is aliasing.
    for name, image, size in [("4000 px page of 3 px strokes -> 336", stroke_page(4000, 3), 336),
                              ("2048 px zone plate -> 336", zone_plate(2048), 336)]:
        rep = aliasing_report(image, size)
        out[name] = rep
        lines += [name] + table([[k, f"{v['rmse_vs_box']:.3f}", f"{v['output_std']:.3f}"] for k, v in rep.items()],
                                ["resampler", "RMSE vs area average", "std of output"]) + [""]

    # 2. The same nominal bicubic resize in two libraries.
    photo = smooth_photo(1024, 768)
    rows, dis = [], {}
    for size in [(336, 252), (224, 168), (1344, 1008)]:
        d = library_disagreement(photo, size)
        dis[f"{size[0]}x{size[1]}"] = d
        for k, v in d.items():
            rows.append([f"1024x768 -> {size[0]}x{size[1]}", k, f"{v['mean_abs']:.2f}", f"{v['max_abs']:.0f}",
                         f"{v['pixels_off_by_more_than_2']:.1%}"])
    out["library_disagreement"] = dis
    lines += ["PIL bicubic vs torch bicubic on a synthetic RGB photo (8-bit levels):"]
    lines += table(rows, ["resize", "torch setting", "mean |diff|", "max |diff|", "values off by > 2"]) + [""]

    # 3. Wrong normalisation constants are an affine change to every input channel.
    rows, mm = [], {}
    for used in NORMS:
        for expected in NORMS:
            if used == expected:
                continue
            g = norm_mismatch(used, expected)
            mm[f"{used} -> {expected}"] = g
            rows.append([used, expected, " ".join(f"{a:.2f}" for a, _ in g), " ".join(f"{b:+.3f}" for _, b in g)])
    out["norm_mismatch"] = mm
    lines += ["Normalised with the wrong constants: x_seen = gain * x_expected + offset (R G B):"]
    lines += table(rows, ["normalised with", "model expects", "gain", "offset"]) + [""]

    # 4. EXIF orientation.
    data = exif_rotated_jpeg(64, 32, 6)
    raw, fixed = load(data, False), load(data, True)
    out["exif"] = {"without_exif_transpose": raw.size, "with_exif_transpose": fixed.size}
    lines += [f"EXIF orientation 6 (phone held upright): Image.open gives {raw.size[0]}x{raw.size[1]}; "
              f"ImageOps.exif_transpose gives {fixed.size[0]}x{fixed.size[1]}.", ""]

    # 5. Alpha flattening.
    chart = transparent_chart()
    rows, alpha = [], {}
    for label, bg in [("convert('RGB'), alpha dropped", None), ("composite on white", (255, 255, 255)),
                      ("composite on black", (0, 0, 0))]:
        c = line_contrast(flatten(chart, bg), chart)
        alpha[label] = c
        rows.append([label, f"{c:.2f}"])
    out["alpha"] = alpha
    lines += ["Black chart lines on a transparent PNG: background minus line brightness (1 = fully visible):"]
    lines += table(rows, ["flattening", "line contrast"])

    print("\n".join(lines))
    write_text("02_preprocessing.txt", lines)
    write_json("02_preprocessing.json", out)


if __name__ == "__main__":
    main()
