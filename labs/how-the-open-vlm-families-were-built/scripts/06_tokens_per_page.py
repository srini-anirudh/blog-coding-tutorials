"""Figure 5 of the post: text tokens per vision token, for real text rendered at different densities.

Network: WikiText-2 (validation split, about 1 MB) and DeepSeek-OCR's tokenizer.json. No model weights.
For each font size we fill a 1024 px page, count its text tokens, and divide by the vision tokens of each
DeepEncoder mode. The paper (as quoted in the post) found about 97% decoding precision below 10x on Fox pages
and about 60% near 20x. We only compute the ratio; precision is not measured.
"""

import _path  # noqa: F401

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

from vlmfam import pages as G
from vlmfam.report import Report

r = Report("06_tokens_per_page")
tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download("deepseek-ai/DeepSeek-OCR", "tokenizer.json"))
lines = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")["text"]
text = G.clean_wikitext(lines)

modes = {m: G.deepencoder_tokens(s) for m, s in G.MODES.items()}
r("DeepEncoder vision tokens per page: " + ", ".join(f"{m} {s} px -> {modes[m]}" for m, s in G.MODES.items()))
r("Text: WikiText-2 validation, rendered on a 1024 x 1024 page, 32 px margins, Pillow's default font.")
r("")
rows, data = [], []
for size in (32, 24, 20, 16, 14, 12, 10):
    fitted, _ = G.fill_page(text, size)
    n = len(tok(fitted, add_special_tokens=False)["input_ids"])
    ratios = {m: n / v for m, v in modes.items()}
    fits10 = [m for m in modes if ratios[m] <= 10]
    rows.append([f"{size} px", len(fitted.split()), n] + [f"{ratios[m]:.1f}x" for m in modes]
                + [fits10[0] if fits10 else "none (tiled mode)"])
    data.append({"font_px": size, "words": len(fitted.split()), "text_tokens": n, "ratios": ratios})
r.table(["font", "words", "text tokens"] + [f"{m} ({v})" for m, v in modes.items()] + ["smallest mode <= 10x"], rows)
r("")
r("Reading: a ratio above 10x is where the paper's precision starts to fall; above 20x it is near 60%.")
r("Dense pages need the larger modes or the tiled Gundam mode, whatever their pixel count, which is the post's")
r("point: set the token budget by how much text a page holds.")
r.save({"modes": modes, "pages": data})
