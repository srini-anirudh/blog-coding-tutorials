"""Sections "What a visual token costs", "Resolution curricula" and Figure 5: the arithmetic, checked.

    python labs/teaching-language-models-to-see/scripts/s5_token_budget.py

1. Recompute every visual-token count the post quotes, and check the native-resolution count against
   Qwen2-VL's real image processor (downloads its 1 MB of config the first time).
2. Build the seven data types of Figure 5 as real token sequences with SmolLM2's tokenizer and C7's processor,
   and measure the share of tokens that carry the loss.
3. The cost arithmetic: image-to-text ratio, training FLOPs, packing and optimizer memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seelab.common import Report  # noqa: E402
from seelab.tokens import (  # noqa: E402
    adam_bytes, ceil_grid_tokens, loss_share, native_tokens, smart_resize, square_tokens, tiled_tokens, train_flops,
)

LM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
QWEN2VL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Example texts for Figure 5. They are written for this lab; only their token counts matter.
ALT_TEXT = "A brown dog catching a frisbee on a beach at sunset."
PAGE = [
    "Every spring the river floods the lower fields. The farmers plant late, after the water has gone, and the "
    "soil it leaves behind is the richest in the valley. The photograph below was taken from the old bridge.",
    "In the picture above, the water has almost reached the first row of houses. The dark line on the walls shows "
    "how high it rose in the worst year on record. Most families now keep a boat tied behind the house, and the "
    "school closes for a week when the river is at its highest. The second photograph shows the same street in June.",
    "By summer the street is dry again, and only the line on the walls is left to show where the water stood.",
]
TRANSCRIPT = " ".join([
    "Annual report of the river authority. Section 3: flood levels and repairs.",
    "The spring flood reached 4.2 metres at the town gauge on 14 April, the third highest level since records began.",
    "Water entered 112 houses on the lower streets, and the main road was closed for six days.",
    "Table 3 lists the repair costs by district. The largest item is the east embankment, where 300 metres of wall",
    "must be rebuilt before next winter. The authority proposes to raise the wall by half a metre and to add two",
    "pumping stations. The total cost is estimated at 2.4 million, of which the regional fund will pay 60 per cent.",
    "Figure 2 shows the gauge readings for the past twenty years. The trend is upward: six of the ten highest",
    "floods occurred in the last decade. The committee recommends a new survey of the upper valley, where the",
    "loss of woodland may have increased the speed of runoff. Section 4 describes the warning system, which now",
    "sends a message to every registered household when the gauge passes 3 metres. In April the first warning",
    "went out nine hours before the water reached the houses, compared with three hours in the previous year.",
    "The authority thanks the volunteers who filled and placed more than 20,000 sandbags during the flood week.",
] * 2)
GROUND_PHRASE = "the man on the left"
GROUND_BOX = "<box>(102,215),(388,940)</box>"
VIDEO_TEXT = "A child rides a red bicycle down the path, stops at the gate and waves at the camera."
QUESTION = "What is written on the sign above the door?"
ANSWER = "The sign says OPEN, in red letters on a white board, and below it the opening hours are listed."


def resolution_table(rep: Report) -> dict:
    rep("## 1. Visual tokens: the post's numbers, recomputed")
    rows = [
        ("336 px, P = 14, no merge (LLaVA-1.5)", square_tokens(336, 14), 576),
        ("448 px, P = 14, 2 x 2 merge", square_tokens(448, 14, 2), 256),
        ("224 px, P = 14 (PaliGemma stage 1)", square_tokens(224, 14), 256),
        ("448 px, P = 14 (PaliGemma stage 2)", square_tokens(448, 14), 1024),
        ("896 px, P = 14 (PaliGemma stage 2)", square_tokens(896, 14), 4096),
        ("Figure 8: page squashed to 224 px", square_tokens(224, 14), 256),
        ("Figure 8: page squashed to 336 px", square_tokens(336, 14), 576),
    ]
    c, r, t6 = tiled_tokens(1000, 1300, max_num=6)
    rows.append((f"Figure 8: InternVL tiling, max 6 tiles -> {c} x {r} + thumbnail", t6, 1792))
    c12, r12, t12 = tiled_tokens(1000, 1300, max_num=12)
    rows.append((f"same page, max 12 tiles -> {c12} x {r12} + thumbnail", t12, None))
    gw, gh, tn = native_tokens(1000, 1300)
    rows.append((f"Figure 8: native (Qwen2-VL rounding) {gw} x {gh}", tn, 1656))
    cw, ch, tc = ceil_grid_tokens(1000, 1300, 14, 2)
    rows.append((f"Figure 7 rule (round the grid up) {cw} x {ch}", tc, None))
    for name, got, post in rows:
        mark = "" if post is None else ("  matches the post" if got == post else f"  post says {post}")
        rep(f"{name:<58} {got:>6,}{mark}")
    rep(f"PaliGemma 896 px tokens / 224 px tokens: {square_tokens(896, 14) / square_tokens(224, 14):.0f}x (post: sixteen times)")
    return {name: {"tokens": got, "post": post} for name, got, post in rows}


def qwen_check(rep: Report) -> dict:
    rep("\n## 1b. Native resolution against Qwen2-VL's real image processor")
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(QWEN2VL_ID)
    except Exception as e:  # offline and not cached
        rep(f"Qwen2-VL image processor unavailable ({type(e).__name__}); skipped")
        return {}
    out = {}
    for w, h in [(1000, 1300), (640, 480), (1920, 1080), (224, 224), (4032, 3024)]:
        img = Image.fromarray(np.full((h, w, 3), 255, np.uint8))
        t, gh, gw = proc(images=img, return_tensors="pt")["image_grid_thw"][0].tolist()
        real = t * gh * gw // 4
        mh, mw = smart_resize(h, w, min_pixels=proc.size["shortest_edge"], max_pixels=proc.size["longest_edge"])
        ours = (mh // 14) * (mw // 14) // 4
        rep(f"{w:>5} x {h:<5} processor grid {gh} x {gw} patches -> {real:>5,} tokens; ours {ours:>5,}"
            f"  {'same' if real == ours else 'DIFFERENT'}")
        out[f"{w}x{h}"] = {"processor": real, "ours": ours}
    return out


def loss_shares(rep: Report) -> dict:
    rep("\n## 2. Figure 5 with real tokens: what share of each sequence carries the loss")
    from transformers import AutoTokenizer

    from tinyvlm.chat import IMAGE_TOKEN
    from tinyvlm.image_processing import TinyVLMImageProcessor
    from tinyvlm.processing import TinyVLMProcessor

    tok = AutoTokenizer.from_pretrained(LM_ID)
    n = lambda s: len(tok(s, add_special_tokens=False).input_ids)  # noqa: E731
    eos = 1  # the end-of-text token is scored too
    page_tokens = native_tokens(1000, 1300)[2]
    rows = {
        "Caption pair": [("image", 576, False), ("text", n(ALT_TEXT) + eos, True)],
        "Interleaved page": [("text", n(PAGE[0]), True), ("image", 256, False), ("text", n(PAGE[1]), True),
                             ("image", 256, False), ("text", n(PAGE[2]) + eos, True)],
        "OCR page": [("image", page_tokens, False), ("text", n(TRANSCRIPT) + eos, True)],
        "Grounded caption": [("image", 576, False), ("text", n(GROUND_PHRASE), True), ("box", n(GROUND_BOX) + eos, True)],
        "Video clip, 4 frames": [("image", 4 * 196, False), ("text", n(VIDEO_TEXT) + eos, True)],
    }
    # The instruction example goes through C7's processor, so the chat template and the label mask are real.
    tok.add_tokens([IMAGE_TOKEN], special_tokens=True)
    proc = TinyVLMProcessor(TinyVLMImageProcessor(size=224), tok, image_seq_len=576)
    ex = proc.build_example(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]},
         {"role": "assistant", "content": ANSWER}], [Image.new("RGB", (224, 224), "white")])
    ids, labels = ex["input_ids"], ex["labels"]
    n_img = sum(int(i == proc.image_token_id) for i in ids)
    n_scored = sum(int(lab != -100) for lab in labels)
    rows["Instruction example"] = [("image", n_img, False), ("prompt", len(ids) - n_img - n_scored, False),
                                   ("text", n_scored, True)]
    rows["Text-only, packed"] = [("text", 2048, True)]
    post = {"Caption pair": 2, "Interleaved page": 26, "OCR page": 19, "Grounded caption": 3,
            "Video clip, 4 frames": 2, "Instruction example": 5, "Text-only, packed": 100}
    out = {}
    for name, segs in rows.items():
        scored, total, share = loss_share(segs)
        layout = " + ".join(f"{k} {v}" for k, v, _ in segs)
        rep(f"{name:<22} {scored:>5,} of {total:>5,} tokens = {100 * share:5.1f}%   (post {post[name]}%)   {layout}")
        out[name] = {"segments": segs, "scored": scored, "total": total, "share": share}
    shares = [out[k]["share"] for k in out if k != "Text-only, packed"]
    rep(f"largest / smallest share among the multimodal rows: {max(shares) / min(shares):.1f}x")
    rep("The instruction row includes the chat template's role markers in the masked prompt.")
    return out


def cost_arithmetic(rep: Report) -> dict:
    rep("\n## 3. Cost arithmetic")
    ratio = 576 / 20
    rep(f"576 image tokens next to a 20-token caption: {ratio:.1f} to 1 (post: near 29 to 1)")
    f_img, f_all = train_flops(7e9, 576), train_flops(7e9, 596)
    rep(f"7B language model, 576 + 20 tokens: {f_all:.2e} training FLOPs per sample, {100 * f_img / f_all:.1f}% on image tokens")
    for length in (2000, 5000):
        k = 8192 // length
        rep(f"{length:,}-token samples in an 8,192 context: {k} fit, {100 * (1 - k * length / 8192):.0f}% left empty")
    frozen, trained = adam_bytes(0, 7e9), adam_bytes(7e9, 0)
    rep(f"7B language model: frozen {frozen / 1e9:.0f} GB, trained {trained / 1e9:.0f} GB before activations (post: 14 and 112)")
    s1, s2 = 1e9 * square_tokens(224, 14), 10e6 * square_tokens(896, 14)
    rep(f"PaliGemma image tokens: 1B examples at 224 px = {s1:.2e}; 10M at 896 px = {s2:.2e} ({100 * s2 / s1:.0f}% of stage 1)")
    return {"image_text_ratio": ratio, "image_flop_share": f_img / f_all, "adam_bytes_7b": [frozen, trained]}


def main() -> None:
    rep = Report("s5_token_budget")
    data = {"resolution": resolution_table(rep), "qwen2vl": qwen_check(rep), "loss_share": loss_shares(rep),
            "cost": cost_arithmetic(rep)}
    rep.save(data)


if __name__ == "__main__":
    main()
