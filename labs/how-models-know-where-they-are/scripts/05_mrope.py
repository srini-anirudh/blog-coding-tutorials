"""Sections 10 and 11: M-RoPE position ids, frequency layouts, absolute time, and positional footprint.

1. Ids for text, image, text. Compared with Hugging Face's Qwen2-VL `get_rope_index` (no download).
2. Text-only M-RoPE equals 1D RoPE exactly.
3. The bookkeeping trap: "max + 1" versus "previous id + 1" versus "token count".
4. Which wavelengths each axis gets under the chunked (Qwen2-VL) and interleaved (Qwen3-VL) layouts.
5. Temporal ids from frame index (Qwen2-VL) versus absolute time (Qwen2.5-VL) at 1 fps and 4 fps.
6. How much of the position range one image uses: flat 1D ids, M-RoPE, V2PE with step 1/16.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posenc import mrope, rope  # noqa: E402
from posenc.report import Report  # noqa: E402


def hf_ids(segments, merge: int = 2):
    """Run transformers' Qwen2VLModel.get_rope_index on the same layout, with a stand-in `self`."""
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

    types, grids = [], []
    for seg in segments:
        if seg[0] == "text":
            types += [0] * seg[1]
        else:
            types += [1] * (seg[1] * seg[2])
            grids.append([1, seg[1] * merge, seg[2] * merge])
    fake = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=merge)))
    fake.get_vision_position_ids = lambda *a, **k: Qwen2VLModel.get_vision_position_ids(fake, *a, **k)
    n = len(types)
    ids, _ = Qwen2VLModel.get_rope_index(
        fake, torch.zeros(1, n, dtype=torch.long), torch.tensor([types]), image_grid_thw=torch.tensor(grids))
    return ids[:, 0]


def main() -> None:
    torch.manual_seed(0)
    rep = Report("05_mrope")
    segs = [("text", 4), ("image", 3, 4), ("text", 3)]

    rep.print("\n1. Text(4) + image(3 rows x 4 cols of merged tokens) + text(3)")
    ids = mrope.position_ids(segs)
    for axis, name in enumerate("thw"):
        rep.print(f"   {name}: {ids[axis].tolist()}")
    try:
        ref = hf_ids(segs)
        same = bool(torch.equal(ids, ref))
        rep.print(f"   identical to transformers Qwen2VLModel.get_rope_index: {same}")
        big = [("text", 7), ("image", 13, 9), ("text", 5), ("image", 4, 20), ("text", 2)]
        rep.print(f"   identical on a longer layout (two images of 13x9 and 4x20): "
                  f"{bool(torch.equal(mrope.position_ids(big), hf_ids(big)))}")
    except Exception as e:  # transformers missing or its API changed
        same = None
        rep.print(f"   transformers comparison skipped: {type(e).__name__}: {e}")
    rep.data["ids"] = ids.tolist()
    rep.data["matches_transformers"] = same

    rep.print("\n2. Text only: M-RoPE (chunked [16, 24, 24]) vs 1D RoPE, head_dim 128, base 1e6")
    f = rope.inv_freq(128, 1e6)
    x = torch.randn(2, 40, 128)
    text_ids = mrope.position_ids([("text", 40)])
    gap = float((mrope.apply_mrope(x, text_ids, f, mrope.axis_of_plane([16, 24, 24], "chunked"))
                 - rope.apply_half(x, torch.arange(40), f)).abs().max())
    rep.print(f"   max |difference| = {gap:.3g}")

    rep.print("\n3. Where the text after an image starts, image of 3 x 4 tokens, 4 text tokens before it")
    rows = []
    for rule in ("max", "last", "count"):
        ids_r = mrope.position_ids(segs, rule=rule)
        after = ids_r[:, -3:]
        image = ids_r[:, 4:16]
        overlap = int((after[1:] <= image[1:].max()).any(0).sum())  # text ids that reuse an image h or w id
        rows.append([rule, str(after[:, 0].tolist()), bool((after[0] == after[1]).all() and (after[1] == after[2]).all()),
                     overlap])
    rep.table(["rule", "first text id (t, h, w)", "t = h = w (pure 1D RoPE)?", "text tokens reusing an image h/w id"], rows)
    rep.data["bookkeeping"] = rows

    rep.print("\n4. Wavelength range (in position ids) per axis, 128-dim head, base 1e6")
    rows = []
    for layout, section in (("chunked", [16, 24, 24]), ("interleaved", [24, 20, 20])):
        rng = mrope.axis_wavelength_range(section, layout, 1e6)
        for axis, (lo, hi) in rng.items():
            rows.append([layout, str(section), axis, lo, hi])
    rep.table(["layout", "section", "axis", "shortest wavelength", "longest wavelength"], rows)
    rep.data["wavelengths"] = rows

    rep.print("\n5. An 8-second clip, temporal patch of 2 frames. Temporal id of the patch that holds second 6.")
    rows = []
    for fps in (1, 4):
        frames = 8 * fps
        patches = frames // 2
        patch_of_second_6 = (6 * fps) // 2
        by_index = mrope.vision_ids(0, patches, 1, 1, 1.0)[0]
        by_time = mrope.vision_ids(0, patches, 1, 1, mrope.qwen25_time_interval(fps))[0]
        rows.append([fps, patches, int(by_index[patch_of_second_6]), int(by_time[patch_of_second_6]), int(by_time[-1])])
    rep.table(["fps", "temporal patches", "id, frame index (Qwen2-VL)", "id, absolute time (Qwen2.5-VL)",
               "last id, absolute time"], rows)
    rep.print("   tokens_per_second = 2 here; the id step per patch is tokens_per_second * 2 / fps.")

    rep.print("\n6. Position range used by one image, 28 px per merged token")
    rows = []
    for side in (448, 1024, 2048):
        fp = mrope.image_footprint(side, side, v2pe_step=1 / 16)
        rows.append([f"{side}x{side}", fp["tokens"], fp["flat_1d"], fp["mrope"], fp["v2pe"]])
    rep.table(["image", "tokens", "flat 1D ids", "M-RoPE ids", "V2PE, step 1/16"], rows)
    rep.data["footprint"] = rows
    rep.save()


if __name__ == "__main__":
    main()
