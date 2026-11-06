"""Post sections: The missing inductive bias (mean attention distance), DeiT and the data problem,
Position: four answers (resizing the table from 224 to 384).

Uses a real trained model, facebook/deit-tiny-distilled-patch16-224 (5.9 M parameters, 12 layers, 3 heads,
a class token and a distillation token), on 64 x 64 tiny-imagenet validation images upsampled to 224.
Needs the Hugging Face Hub once (about 24 MB of weights, 15 MB of images).

    python labs/an-image-is-a-sentence/scripts/06_pretrained_deit.py [--images 1000]
Saves results/06_pretrained_deit.txt and .json.
"""

import argparse

import _setup  # noqa: F401
import torch
import torch.nn.functional as F
from transformers import DeiTForImageClassificationWithTeacher

from vitfamily.c03 import be_polite
from vitfamily.data import tiny_imagenet_val
from vitfamily.patch import resize_pos_table
from vitfamily.report import Report
from vitfamily.stats import mean_attention_distance

p = argparse.ArgumentParser()
p.add_argument("--images", type=int, default=1000)
p.add_argument("--batch", type=int, default=50)
args = p.parse_args()
be_polite(8)
log = Report("06_pretrained_deit")

NAME = "facebook/deit-tiny-distilled-patch16-224"
model = DeiTForImageClassificationWithTeacher.from_pretrained(NAME, attn_implementation="eager").eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
x_all, _ = tiny_imagenet_val()
x_all = x_all[: args.images]


def prep(x: torch.Tensor, size: int) -> torch.Tensor:
    x = F.interpolate(x.float() / 255, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - MEAN) / STD


log(f"model {NAME}; {len(x_all)} tiny-imagenet validation images, 64 px upsampled bicubically")

# 1. Mean attention distance, every head of every layer, at 224 (14 x 14 patches of 16 px).
layers, heads = model.config.num_hidden_layers, model.config.num_attention_heads
mad = torch.zeros(layers, heads)
cos_layers = torch.zeros(layers + 1)
logits_cls, logits_dist, preds224 = [], [], []
with torch.no_grad():
    for i in range(0, len(x_all), args.batch):
        out = model.deit(prep(x_all[i : i + args.batch], 224), output_attentions=True, output_hidden_states=True)
        w = len(x_all[i : i + args.batch]) / len(x_all)
        for layer, a in enumerate(out.attentions):  # (B, heads, 198, 198)
            mad[layer] += w * mean_attention_distance(a, side=14, patch_px=16, n_special=2)
        for layer, h in enumerate(out.hidden_states):  # layer 0 is the embedding output
            cos_layers[layer] += w * F.cosine_similarity(h[:, 0], h[:, 1]).mean()
        seq = out.last_hidden_state  # after the final LayerNorm
        lc, ld = model.cls_classifier(seq[:, 0]), model.distillation_classifier(seq[:, 1])
        logits_cls.append(lc), logits_dist.append(ld), preds224.append(((lc + ld) / 2).argmax(-1))
logits_cls, logits_dist, preds224 = torch.cat(logits_cls), torch.cat(logits_dist), torch.cat(preds224)

log("\n== Mean attention distance per head, in pixels, 224 px input (uniform attention would give 116.5 px) ==")
for layer in range(layers):
    vals = ", ".join(f"{v:5.1f}" for v in mad[layer].tolist())
    log(f"layer {layer + 1:>2}: heads [{vals}]  spread {mad[layer].max() - mad[layer].min():5.1f}")
log.data["mean_attention_distance_px"] = [[round(v, 2) for v in r] for r in mad.tolist()]

# 2. The two special tokens.
emb = model.deit.embeddings
tok_cos = F.cosine_similarity(emb.cls_token[0, 0], emb.distillation_token[0, 0], dim=0).item()
disagree = (logits_cls.argmax(-1) != logits_dist.argmax(-1)).float().mean().item()
log("\n== Class token vs distillation token ==")
log(f"cosine of the learned input tokens (parameters): {tok_cos:.3f}")
log("cosine of their hidden states, after embedding (0) and each layer: "
    + ", ".join(f"{c:.2f}" for c in cos_layers.tolist()))
log(f"the two heads' top-1 predictions disagree on {100 * disagree:.1f}% of images")
log.data.update({"cos_input_tokens": tok_cos, "cos_by_layer": cos_layers.tolist(), "head_disagreement": disagree})

# 3. Resize the position table to 384 (24 x 24 patches), correctly and with the special-row bug.
log("\n== Position table 224 -> 384: (1, 198, 192) -> (1, 578, 192) ==")
table = emb.position_embeddings.data.clone()
agreement = {}


def predict_384(**kw) -> torch.Tensor:
    return torch.cat([model(pixel_values=prep(x_all[i : i + args.batch], 384), **kw).logits.argmax(-1)
                      for i in range(0, len(x_all), args.batch)])  # fmt: skip


with torch.no_grad():
    ref = predict_384(interpolate_pos_encoding=True)  # transformers' own resize, the reference
    ours = resize_pos_table(table, 24, n_special=2, hold_out=True)
    agreement["transformers interpolate_pos_encoding"] = (ref == preds224).float().mean().item()
    log(f"{'transformers interpolate_pos_encoding':<40} top-1 agrees with the 224 px prediction on "
        f"{100 * agreement['transformers interpolate_pos_encoding']:.1f}% of images")
    emb.image_size = emb.patch_embeddings.image_size = (384, 384)
    variants = {"ours, special rows held out": ours,
                "bug: special rows resized as patches": resize_pos_table(table, 24, n_special=2, hold_out=False)}  # fmt: skip
    for name, new_table in variants.items():
        emb.position_embeddings = torch.nn.Parameter(new_table)
        preds = predict_384()
        agreement[name] = (preds == preds224).float().mean().item()
        log(f"{name:<40} top-1 agrees with the 224 px prediction on {100 * agreement[name]:.1f}% of images; "
            f"with transformers' resize on {100 * (preds == ref).float().mean().item():.1f}%")
log.data["agreement_384_vs_224"] = agreement
log.save()
