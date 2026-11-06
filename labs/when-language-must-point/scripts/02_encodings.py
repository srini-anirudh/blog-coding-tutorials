"""One box, eight encodings: write, count, parse back, and parse with the wrong convention (post table and Nuance)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OFFLINE = os.environ.get("HF_HUB_OFFLINE") == "1"   # set it to count with cached tokenizers only

from pointlab.coords import BY_NAME, CONVENTIONS, QWEN25_SIZE, digit_split_count, smart_resize  # noqa: E402
from pointlab.geometry import box_iou  # noqa: E402
from pointlab.report import Report  # noqa: E402

R = Report("02_encodings")
BOX = (0.268, 0.372, 0.584, 0.811)   # the dog in the post: 26.8% to 58.4% of the width, 37.2% to 81.1% of the height


def load_tokenizers():
    """Real BPE tokenizers that split numbers into single digits, if they are in the local Hugging Face cache."""
    out = {}
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return out
    for name in ("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B-Instruct"):
        try:
            out[name.split("/")[1]] = AutoTokenizer.from_pretrained(name, local_files_only=OFFLINE)
        except Exception as e:  # no network and not cached: the digit-split count still runs
            R(f"(tokenizer {name} not available offline: {type(e).__name__})")
    return out


def real_count(tok, conv, text):
    """Token count with the convention's location tags added to the vocabulary, as those models do."""
    if conv.special_tokens:
        tok = tok.__class__.from_pretrained(tok.name_or_path, local_files_only=OFFLINE)
        tok.add_tokens(list(conv.special_tokens), special_tokens=True)
    return len(tok(text, add_special_tokens=False)["input_ids"])


toks = load_tokenizers()
rows, data = [], {}
for c in CONVENTIONS:
    s = c.encode(BOX)
    back = c.parse(s)
    err = max(abs(a - b) for a, b in zip(back, BOX)) if not c.name.startswith("Molmo") else 0.0
    counts = [str(real_count(t, c, s)) for t in toks.values()]
    rows.append([c.name, s, digit_split_count(s, c.special_tokens), *counts, f"{err * 100:.2f}%"])
    data[c.name] = {"text": s, "digit_split_tokens": digit_split_count(s, c.special_tokens),
                    **{f"tokens_{k}": int(v) for k, v in zip(toks, counts)}, "max_roundtrip_error": err}
R("One box, eight encodings (the strings match the post's table)")
R.table(["system", "what the model writes", "tokens (digit split)", *[f"tokens ({k})" for k in toks],
         "worst edge error after parsing, share of image"], rows)
R("Molmo writes a point (the box centre), so it has no edges to compare.")

# ---- the same string, the wrong parser ---------------------------------------------------------
R("")
R("Parsing with the wrong convention: no exception, a valid-looking box, in the wrong place")
wrong = []
pg = BY_NAME["PaliGemma"].encode(BOX)
as_xy = tuple(int(v) / 1024 for v in __import__("re").findall(r"<loc(\d{4})>", pg))
wrong.append(["PaliGemma string read x-first", pg[:38], f"{box_iou(as_xy, BOX):.3f}"])
q25 = BY_NAME["Qwen2.5-VL"].encode(BOX)
nums = __import__("json").loads(q25)["bbox_2d"]
as_grid = tuple(v / 1000 for v in nums)
wrong.append(["Qwen2.5-VL pixels read as a 0-1000 grid", str(nums), f"{box_iou(as_grid, BOX):.3f}"])
q3 = [round(v * 1000) for v in BOX]
as_px = (q3[0] / QWEN25_SIZE[0], q3[1] / QWEN25_SIZE[1], q3[2] / QWEN25_SIZE[0], q3[3] / QWEN25_SIZE[1])
wrong.append(["Qwen3-VL 0-1000 grid read as Qwen2.5-VL pixels", str(q3), f"{box_iou(as_px, BOX):.3f}"])
R.table(["mistake", "numbers", "IoU with the true box"], wrong)
R("(IoU is computed in normalised coordinates; a box that leaves the image is not clipped here.)")
data["wrong_parser_iou"] = {r[0]: float(r[2]) for r in wrong}

# ---- what "actual size" means for Qwen2.5-VL ---------------------------------------------------
R("")
R("Qwen2.5-VL coordinates are pixels of the resized input, not of the original file")
cases = [("868 x 644 photo (post example)", 644, 868, {}),
         ("3840 x 2160 screenshot, default max_pixels", 2160, 3840, {}),
         ("3840 x 2160 screenshot, max_pixels = 1280 * 28 * 28", 2160, 3840, {"max_pixels": 1280 * 28 * 28})]
rs = []
for name, h, w, kw in cases:
    rh, rw = smart_resize(h, w, **kw)
    px = [round(BOX[0] * rw), round(BOX[1] * rh), round(BOX[2] * rw), round(BOX[3] * rh)]
    orig = [round(BOX[0] * w), round(BOX[1] * h), round(BOX[2] * w), round(BOX[3] * h)]
    rs.append([name, f"{rw} x {rh}", str(px), str(orig), str([round(v * 1000) for v in BOX])])
R.table(["image", "resized input", "Qwen2.5-VL writes (resized px)", "original-file px", "Qwen3-VL writes (0-1000)"], rs)
R("Reading the Qwen2.5-VL numbers as original-file pixels puts the box in the wrong place. With a 1316 x 728")
R(f"input and a 3840 x 2160 original, every coordinate comes out too small by a factor of {3840 / 1316:.2f}.")

R.save(data)
