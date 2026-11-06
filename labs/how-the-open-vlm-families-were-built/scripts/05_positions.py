"""Positions across Qwen-VL generations, and InternVL3's V2PE.

Offline. The rope settings below are copied from the Hub configs of Qwen2-VL-7B, Qwen2.5-VL-7B and Qwen3-VL-8B
(mrope_section, rope_theta, tokens_per_second). Part 1 checks `vlmfam.positions.mrope_ids` against the HF
get_rope_index of Qwen2-VL and Qwen2.5-VL, and `axis_per_freq` against Qwen3-VL's frequency recomposition.
"""

import _path  # noqa: F401

import math
import random
from types import SimpleNamespace

import torch

from vlmfam import positions as P
from vlmfam import tokens as T
from vlmfam.report import Report

QWEN2 = {"section": [16, 24, 24], "theta": 1e6, "head_dim": 128}  # Qwen2-VL-7B and Qwen2.5-VL-7B
QWEN3 = {"section": [24, 20, 20], "theta": 5e6, "head_dim": 128}  # Qwen3-VL-8B, interleaved
TOKENS_PER_SECOND = 2  # Qwen2.5-VL-7B vision_config.tokens_per_second

r = Report("05_positions")


# ---------------------------------------------------------------- part 1: parity with HF
def hf_rope_index(cls, segments, merge=2, tps=TOKENS_PER_SECOND):
    """Call HF's get_rope_index with only the config fields it reads (no model weights)."""
    types, grids_img, grids_vid, secs = [], [], [], []
    for seg in segments:
        if seg[0] == "text":
            types += [0] * seg[1]
            continue
        t, h, w = seg[1]
        n = t * h * w // merge ** 2
        types += [1 if seg[0] == "image" else 2] * n
        (grids_img if seg[0] == "image" else grids_vid).append([t, h, w])
        if seg[0] == "video":
            secs.append(seg[2] / tps)  # interval = tps * seconds per temporal patch
    cfg = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=merge, tokens_per_second=tps))
    fake = SimpleNamespace(config=cfg, get_vision_position_ids=lambda *a, **k: cls.get_vision_position_ids(None, *a, **k))
    ids = torch.zeros(1, len(types), dtype=torch.long)
    kw = {"second_per_grid_ts": torch.tensor(secs)} if "2_5" in cls.__module__ and secs else {}
    pos, _ = cls.get_rope_index(fake, ids, torch.tensor([types]), torch.tensor(grids_img) if grids_img else None,
                                torch.tensor(grids_vid) if grids_vid else None, **kw)
    return pos[:, 0]


from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel  # noqa: E402
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding  # noqa: E402

rng = random.Random(0)
trials, agree = 0, 0
for _ in range(40):
    segs = []
    for _ in range(rng.randint(1, 4)):
        segs.append(("text", rng.randint(1, 7)))
        kind = rng.choice(["image", "video"])
        grid = (1 if kind == "image" else 2 * rng.randint(1, 4), 2 * rng.randint(1, 5), 2 * rng.randint(1, 5))
        segs.append((kind, grid) if kind == "image" else (kind, grid, 1))
    segs.append(("text", 3))
    for cls, interval in ((Qwen2VLModel, 1), (Qwen2_5_VLModel, rng.choice([1, 2, 4]))):
        s = [seg if seg[0] != "video" else (seg[0], seg[1], interval) for seg in segs]
        trials += 1
        agree += torch.equal(P.mrope_ids(s), hf_rope_index(cls, s))
r(f"Part 1. mrope_ids == HF get_rope_index on {agree} of {trials} random text/image/video sequences")
freq = torch.stack([torch.full((1, 1, 64), float(a)) for a in range(3)])  # axis id in every slot
hf_axes = Qwen3VLTextRotaryEmbedding.recomposition_frequencies(SimpleNamespace(mrope_section=QWEN3["section"]),
                                                                freq)[0, 0, :64].long().tolist()
same = hf_axes == P.axis_per_freq(QWEN3["section"], interleaved=True)
r(f"        axis_per_freq(interleaved) == Qwen3-VL recomposition_frequencies: {same}")

# ---------------------------------------------------------------- part 2: what an image costs in positions
r("")
r("Part 2. Position range used by one image (text before and after: 10 tokens each)")
rows = []
for name, w, h in T.PRESETS:
    c = T.qwen2_vl(w, h)
    t_, gh, gw = c.parts["grid_thw"]
    ids = P.mrope_ids([("text", 10), ("image", (t_, gh, gw)), ("text", 10)])
    rows.append([name, c.tokens, c.tokens + 20, P.positions_used(ids)])
r.table(["image", "image tokens", "1D positions used", "M-RoPE positions used"], rows)

# ---------------------------------------------------------------- part 3: video time
r("")
r("Part 3. One 60 s clip, 448 x 448 frames, two frames per temporal patch. Temporal ids of the video tokens:")
rows = []
for fps in (1, 2, 4):
    patches = 60 * fps // 2
    sec_per_patch = 2 / fps
    q2 = P.mrope_ids([("video", (patches, 32, 32), 1)])
    q25 = P.mrope_ids([("video", (patches, 32, 32), TOKENS_PER_SECOND * sec_per_patch)])
    rows.append([f"{fps} fps", patches, f"0..{int(q2[0].max())}", f"0..{int(q25[0].max())}",
                 f"{int(q25[0].unique()[1])}" if patches > 1 else "-"])
r.table(["sampling", "temporal patches", "Qwen2-VL time ids", "Qwen2.5-VL time ids", "Qwen2.5 step"], rows)
r("  Qwen2-VL counts patches, so the same clip gets a different range at each frame rate. Qwen2.5-VL ties the id")
r(f"  to seconds ({TOKENS_PER_SECOND} ids per second), so 60 s always spans about {60 * TOKENS_PER_SECOND} ids.")
hour = P.mrope_ids([("video", (1800, 2, 2), TOKENS_PER_SECOND * 2.0)])  # 1 h at 1 fps = 1,800 patches of 2 s
r(f"  A 1-hour video at 1 fps reaches time id {int(hour[0].max())}; Qwen3-VL instead writes '<t seconds>' as text")
r("  and keeps the time axis small (post: 'large and sparse on long videos').")

# ---------------------------------------------------------------- part 4: frequency allocation
r("")
r("Part 4. Which rotary frequencies each axis gets (head_dim 128 -> 64 frequencies, fastest first)")
rows = []
for name, cfg, inter in (("Qwen2-VL / 2.5-VL", QWEN2, False), ("Qwen3-VL", QWEN3, True)):
    axes = P.axis_per_freq(cfg["section"], inter)
    inv = [cfg["theta"] ** (-2 * i / cfg["head_dim"]) for i in range(cfg["head_dim"] // 2)]
    for a, label in enumerate("THW"):
        idx = [i for i, x in enumerate(axes) if x == a]
        waves = [2 * math.pi / inv[i] for i in idx]
        rows.append([name, label, len(idx), f"{idx[0]}..{idx[-1]}", f"{min(waves):.1f}", f"{max(waves):.3g}"])
r.table(["model", "axis", "frequencies", "index range", "shortest wavelength", "longest wavelength"], rows)
r("  Contiguous blocks give time only fast frequencies and width only slow ones. Interleaving gives every axis")
r("  both, which is what the post means by 'each cover both high and low frequencies'.")

# ---------------------------------------------------------------- part 5: V2PE
r("")
r("Part 5. V2PE (InternVL3): positions used by 4 letter-scan images (13 tiles x 256 tokens each) plus 400 text tokens")
n_img = T.internvl(1700, 2200).tokens
segs = [("text", 100), ("image", n_img)] * 4 + [("text", 0)]
rows = []
for delta in (1, 1 / 4, 1 / 16, 1 / 64):
    pos = P.v2pe_positions(segs, delta)
    rows.append([f"{delta:g}", len(pos), f"{pos.max().item() + 1:.0f}"])
r.table(["visual step delta", "tokens", "position range used"], rows)
r("  The delta values are illustrative. Text keeps its step of 1, so the language model's sense of distance in")
r("  text is unchanged while images take a fraction of the position range.")
r.save({"rope_parity": [agree, trials], "qwen3_interleave_parity": same})
assert agree == trials and same
