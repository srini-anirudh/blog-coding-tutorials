"""Sections 08, 11 and 12 on two small public checkpoints. Downloads them from the Hugging Face Hub on first run.

Part A. SmolLM2-135M (Pre-LN, RMSNorm, 30 layers) on Tiny Shakespeare text:
  residual-stream norm by layer, the largest activations (massive activations) and where they sit,
  and the loss increase when each single layer is removed (the curse of depth).
Part B. SmolVLM-256M-Instruct (SigLIP-style tower, pixel-shuffle + linear connector, SmolLM2 backbone):
  the norm of visual tokens before and after the connector against the norm of text embeddings (Fig 10).
Part C. SmolVLM's embedding table against SmolLM2-135M's: did vision-language training inflate it,
  and how large are the image-boundary tokens? (The post reports this for Qwen2-VL; this is a different, smaller model.)
"""

import argparse
import io

import _setup  # noqa: F401
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from normgeo.report import Report, shakespeare

p = argparse.ArgumentParser()
p.add_argument("--windows", type=int, default=8)
p.add_argument("--seq", type=int, default=256)
p.add_argument("--images", type=int, default=16)
args = p.parse_args()
torch.manual_seed(0)
LM, VLM = "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolVLM-256M-Instruct"

with Report("10_real_models") as say:
    tok = AutoTokenizer.from_pretrained(LM)
    lm = AutoModelForCausalLM.from_pretrained(LM, torch_dtype=torch.float32).eval()
    text = shakespeare()
    ids = tok(text[int(0.9 * len(text)) :], return_tensors="pt").input_ids[0]
    batch = torch.stack([ids[i * args.seq : (i + 1) * args.seq] for i in range(args.windows)])
    say(f"Part A. {LM}, fp32, {args.windows} windows x {args.seq} tokens of Tiny Shakespeare (no BOS added).")
    with torch.no_grad():
        out = lm(batch, labels=batch, output_hidden_states=True, use_cache=False)
    base_loss = out.loss.item()
    hs = out.hidden_states  # embeddings, layers 1-29, then layer 30 with the final RMSNorm already applied (HF convention)
    say(f"  loss {base_loss:.3f}. Stream norm per token (mean over positions 1+), norm at position 0, largest |activation|.")
    say(f"  {'after layer':>11}{'mean ||h||':>12}{'||h|| pos 0':>13}{'max |h|':>10}{'at (pos, channel)':>20}{'median |h|':>12}")
    layers = []
    for l, h in enumerate(hs):
        a = h.abs()
        flat = a.flatten().argmax().item()
        _, pos, ch = torch.unravel_index(torch.tensor(flat), a.shape)
        row = {"layer": l, "mean_norm": h[:, 1:].norm(dim=-1).mean().item(), "pos0_norm": h[:, 0].norm(dim=-1).mean().item(),
               "max_abs": a.max().item(), "max_pos": int(pos), "max_channel": int(ch), "median_abs": a.median().item()}
        layers.append(row)
        if l % 3 == 0 or l == len(hs) - 1:
            label = f"{l} + norm" if l == len(hs) - 1 else str(l)
            say(f"  {label:>11}{row['mean_norm']:>12.2f}{row['pos0_norm']:>13.2f}{row['max_abs']:>10.1f}"
                f"{f'({pos}, {ch})':>20}{row['median_abs']:>12.3f}")
    say("  The last row is after the final RMSNorm, which brings the stream back to a bounded scale before the head.")

    say()
    say("  Loss increase when one layer is removed:")
    full = lm.model.layers
    drops = []
    for l in range(len(full)):
        lm.model.layers = torch.nn.ModuleList([m for i, m in enumerate(full) if i != l])
        lm.config.num_hidden_layers = len(full) - 1
        with torch.no_grad():
            drops.append(lm(batch, labels=batch, use_cache=False).loss.item() - base_loss)
    lm.model.layers, lm.config.num_hidden_layers = full, len(full)
    say("  " + " ".join(f"{l}:{d:+.2f}" for l, d in enumerate(drops)))
    third = len(drops) // 3
    say(f"  mean increase, layers 0-{third - 1}: {sum(drops[:third]) / third:.3f};  layers {third}-{2 * third - 1}: "
        f"{sum(drops[third:2 * third]) / third:.3f};  layers {2 * third}-{len(drops) - 1}: {sum(drops[2 * third:]) / (len(drops) - 2 * third):.3f}")
    lm_emb = lm.model.embed_tokens.weight.detach().clone()
    text_norm = lm_emb[batch.flatten()].norm(dim=-1).mean().item()
    del lm

    say()
    proc = AutoProcessor.from_pretrained(VLM)
    vlm = AutoModelForImageTextToText.from_pretrained(VLM, torch_dtype=torch.float32).eval()
    path = hf_hub_download("uoft-cs/cifar10", "plain_text/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path).iloc[: args.images]
    imgs = [Image.open(io.BytesIO(b["bytes"])).convert("RGB") for b in df["img"]]
    px = proc.image_processor(imgs, do_image_splitting=False, return_tensors="pt")
    pv = px.pixel_values.flatten(0, 1)
    with torch.no_grad():
        vis = vlm.model.vision_model(pixel_values=pv).last_hidden_state
        conn = vlm.model.connector(vis)
    emb = vlm.model.text_model.embed_tokens.weight.detach()
    vlm_text = emb[batch.flatten()].norm(dim=-1).mean().item()
    say(f"Part B. {VLM}, {args.images} CIFAR-10 test images (upscaled by the processor, no splitting).")
    say(f"  vision tower output    {tuple(vis.shape[1:])} per image, mean token norm {vis.norm(dim=-1).mean():.2f}")
    say(f"  after the connector    {tuple(conn.shape[1:])} per image, mean token norm {conn.norm(dim=-1).mean():.2f}")
    say(f"  text token embeddings  mean norm {vlm_text:.3f} (the Part A text, in SmolVLM's own table)")
    say(f"  ratio visual / text after the connector: {conn.norm(dim=-1).mean().item() / vlm_text:.1f}x")

    say()
    shared = lm_emb.shape[0]
    n_lm, n_vlm = lm_emb.norm(dim=-1), emb[:shared].norm(dim=-1)
    cos = F.cosine_similarity(lm_emb, emb[:shared], dim=-1)
    say(f"Part C. Embedding tables. {shared} rows shared with SmolLM2-135M, {emb.shape[0] - shared} rows added by SmolVLM.")
    say(f"  shared rows: cosine(SmolLM2 row, SmolVLM row) median {cos.median():.3f}")
    say(f"  mean row norm: SmolLM2-135M {n_lm.mean():.3f}, SmolVLM {n_vlm.mean():.3f}. Max: {n_lm.max():.3f} vs {n_vlm.max():.3f}")
    all_norms = emb.norm(dim=-1)
    order = all_norms.argsort(descending=True)
    rank = {int(t): r for r, t in enumerate(order)}
    for t in ["<fake_token_around_image>", "<global-img>", "<image>", "<row_1_col_1>"]:
        tid = proc.tokenizer.convert_tokens_to_ids(t)
        say(f"  {t:<26} id {tid:>6}  norm {all_norms[tid]:.3f}  rank {rank[tid] + 1} of {emb.shape[0]} by norm")
    say.json({"lm_loss": base_loss, "layers": layers, "drop_cost": drops, "vision_norm": vis.norm(dim=-1).mean().item(),
              "connector_norm": conn.norm(dim=-1).mean().item(), "text_norm_vlm": vlm_text, "text_norm_lm": text_norm,
              "emb_mean_norm_lm": n_lm.mean().item(), "emb_mean_norm_vlm": n_vlm.mean().item()})
