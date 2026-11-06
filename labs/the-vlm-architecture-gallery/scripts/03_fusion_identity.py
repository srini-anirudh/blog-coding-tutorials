"""Question 3, joining: what each fusion point does to a pretrained LM before any training.

The post makes three structural claims that can be checked exactly at init:
  - gated cross-attention is zero-initialised, so the model is an exact copy of the frozen LM,
    even with an image attached;
  - a visual expert leaves text positions on the original weights, so text-only behaviour is
    identical; with an image, text logits change only through shared attention;
  - a full visual expert doubles the decoder's projection parameters; PLoRA adds a small fraction.
"""

import _common
import torch

from gallery.data import GlyphPages
from gallery.fusion import TinyLM
from gallery.model import TinyVLM
from gallery.train import fresh_copy, pretrain_lm

_common.threads()
rep = _common.Report("03_fusion_identity")
lm = pretrain_lm(seed=0, steps=1500, cache=_common.RUNS / "lm_seed0.pt")
lm_params = sum(p.numel() for p in lm.parameters())
data = GlyphPages()
g = torch.Generator().manual_seed(5)
img, q, _ = data.visual_batch(8, "fine", g)
text_ids, _ = data.text_batch(8, "add", g)

with torch.no_grad():
    base_q = lm(q)                       # the LM reading only the question
    base_text = lm(text_ids)
rep(f"pretrained tiny LM: {lm_params:,} parameters")
rep(f"{'fusion':<22}{'added params':>13}{'x LM':>7}{'text-only identical':>21}{'with image: max |dlogit| at question':>40}")
rows = []
for label, kw in [("prefix", dict(fusion="prefix")), ("gated xattn", dict(fusion="xattn")),
                  ("visual expert", dict(fusion="expert")), ("PLoRA r=8", dict(fusion="expert", expert_rank=8)),
                  ("encoder-free", dict(fusion="encfree"))]:
    torch.manual_seed(0)
    vlm = TinyVLM(fresh_copy(lm), connector="mlp", **kw).eval()
    added = sum(p.numel() for p in vlm.lm.parameters()) - lm_params
    with torch.no_grad():
        same_text = torch.equal(vlm.lm(text_ids), base_text)
        if kw["fusion"] == "xattn":
            # Flamingo: the text stream is the question alone, and it now attends to the image
            with_img = vlm.lm.forward_embeds(vlm.lm.embed(q), mem=vlm.visual_tokens(img))
            delta = (with_img - base_q).abs().max().item()
        else:
            # prefix-style: compare the question positions with and without the image in front
            logits = vlm(img, q)
            delta = (logits - base_q[:, -1]).abs().max().item()
    rep(f"{label:<22}{added:>13,}{added / lm_params:>7.2f}{str(same_text):>21}{delta:>40.2e}")
    rows.append({"fusion": label, "added_lm_params": added, "text_only_identical": same_text,
                 "max_logit_change_with_image": delta})
rep("")
rep("gated xattn changes nothing even with an image attached (tanh(0) = 0): the frozen LM is reproduced exactly.")
rep("prefix-style fusions change the question logits as soon as an image is in front: the LM attends to it.")
rep.save({"lm_params": lm_params, "rows": rows})
