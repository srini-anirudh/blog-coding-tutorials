"""The connector section of the post: four patterns, their token counts, and their real sizes.

Part 1 (offline): each from-scratch connector against the Hugging Face module it reproduces, at toy size.
Part 2 (offline): what each connector does to the 1,024 patches of one 448 px tile.
Part 3 (network, config.json only): connector parameters predicted from each config, against the count read from
the checkpoint headers by scripts/03_param_split.py (run that first).
"""

import _path  # noqa: F401

import json
import sys

import torch

from vlmfam import connectors as C
from vlmfam.report import RESULTS, Report

torch.manual_seed(0)
r = Report("04_connectors")


def copy(dst, src_state: dict) -> None:
    dst.load_state_dict(src_state, strict=True)


# ---------------------------------------------------------------- part 1: parity with HF
from transformers import Gemma3Config, InternVLConfig, LlavaConfig  # noqa: E402
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector  # noqa: E402
from transformers.models.idefics3.modeling_idefics3 import Idefics3Connector  # noqa: E402
from transformers.models.internvl.modeling_internvl import InternVLModel, InternVLMultiModalProjector  # noqa: E402
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector  # noqa: E402
from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchMerger as HFPatchMerger  # noqa: E402

r("Part 1. From-scratch connector against the HF module (random weights, toy size): max |difference|")
checks = []

x = torch.randn(2, 64, 12)  # 8 x 8 patches, d = 12
checks.append(("pixel shuffle (C7) vs Idefics3Connector.pixel_shuffle",
               C.pixel_shuffle(x, 2), Idefics3Connector.pixel_shuffle(None, x, 2)))
g = x.reshape(2, 8, 8, 12)
checks.append(("InternVL pixel shuffle vs InternVLModel.pixel_shuffle",
               C.internvl_pixel_shuffle(g), InternVLModel.pixel_shuffle(None, g, 0.5)))
same_set = torch.equal(C.pixel_shuffle(x, 2).sort(-1).values,
                       C.internvl_pixel_shuffle(g).reshape(2, 16, 48).sort(-1).values)

hf = HFPatchMerger(dim=16, context_dim=12, spatial_merge_size=2)
ours = C.PatchMerger(12, 16)
copy(ours, {"ln.weight": hf.ln_q.weight, "ln.bias": hf.ln_q.bias, "fc1.weight": hf.mlp[0].weight,
            "fc1.bias": hf.mlp[0].bias, "fc2.weight": hf.mlp[2].weight, "fc2.bias": hf.mlp[2].bias})
p = torch.randn(64, 12)
checks.append(("Qwen2-VL PatchMerger", ours(p), hf(p)))

lcfg = LlavaConfig(vision_config={"model_type": "clip_vision_model", "hidden_size": 12},
                   text_config={"model_type": "llama", "hidden_size": 16, "num_attention_heads": 4, "num_key_value_heads": 4})
hf = LlavaMultiModalProjector(lcfg)
ours = C.MLPConnector(12, 16)
copy(ours, {"fc1.weight": hf.linear_1.weight, "fc1.bias": hf.linear_1.bias, "fc2.weight": hf.linear_2.weight,
            "fc2.bias": hf.linear_2.bias})
checks.append(("LLaVA MLP projector", ours(x), hf(x)))

icfg = InternVLConfig(vision_config={"hidden_size": 12}, text_config={"model_type": "qwen2", "hidden_size": 16, "num_attention_heads": 4, "num_key_value_heads": 4})
hf = InternVLMultiModalProjector(icfg)
ours = C.MLPConnector(48, 16, pre_norm=True)
copy(ours, {"norm.weight": hf.layer_norm.weight, "norm.bias": hf.layer_norm.bias, "fc1.weight": hf.linear_1.weight,
            "fc1.bias": hf.linear_1.bias, "fc2.weight": hf.linear_2.weight, "fc2.bias": hf.linear_2.bias})
xs = C.internvl_pixel_shuffle(g).reshape(2, 16, 48)
checks.append(("InternVL projector (LN + MLP)", ours(xs), hf(xs)))

gcfg = Gemma3Config(vision_config={"hidden_size": 12, "image_size": 56, "patch_size": 7},
                    text_config={"hidden_size": 16, "num_attention_heads": 4, "num_key_value_heads": 4}, mm_tokens_per_image=4)
hf = Gemma3MultiModalProjector(gcfg)
with torch.no_grad():
    hf.mm_input_projection_weight.normal_()
    hf.mm_soft_emb_norm.weight.normal_()
ours = C.AvgPoolConnector(12, 16, patches_per_side=8, tokens=4, eps=gcfg.vision_config.layer_norm_eps)
copy(ours, {"norm_weight": hf.mm_soft_emb_norm.weight, "proj": hf.mm_input_projection_weight})
checks.append(("Gemma 3 average pool + RMSNorm + matrix", ours(x), hf(x)))

worst = 0.0
for name, a, b in checks:
    d = (a - b).abs().max().item()
    worst = max(worst, d)
    r(f"  {name:<56} {tuple(a.shape)}  {d:.2e}")
r(f"  Idefics3 and InternVL shuffles hold the same values in a different channel order: {same_set}")

# ---------------------------------------------------------------- part 2: tokens for one tile
r("")
r("Part 2. One 448 px tile = 32 x 32 = 1,024 patches of width 1,152 (SigLIP So400m). LLM width 3,584.")
dv, dt, tile = 1152, 3584, torch.randn(1, 1024, 1152)
rows = []
mods = [
    ("MLP on every patch (LLaVA)", lambda z: C.MLPConnector(dv, dt)(z), C.MLPConnector(dv, dt)),
    ("2 x 2 merge + MLP (Qwen2-VL)", lambda z: C.PatchMerger(dv, dt)(z[0]), C.PatchMerger(dv, dt)),
    ("pixel shuffle x4 + MLP (InternVL)", lambda z: C.MLPConnector(4 * dv, dt)(C.pixel_shuffle(z, 2)),
     C.MLPConnector(4 * dv, dt)),
    ("pixel shuffle x16 + linear (SmolVLM)", lambda z: torch.nn.Linear(16 * dv, dt, bias=False)(C.pixel_shuffle(z, 4)),
     torch.nn.Linear(16 * dv, dt, bias=False)),
    ("4 x 4 average pool (Gemma 3)", lambda z: C.AvgPoolConnector(dv, dt, 32, 64)(z), C.AvgPoolConnector(dv, dt, 32, 64)),
    ("64 learned queries (MiniCPM-V)", lambda z: C.QueryResampler(dv, dt, 64)(z), C.QueryResampler(dv, dt, 64)),
]
with torch.no_grad():
    for name, f, m in mods:
        out = f(tile)
        n = out.shape[-2] if out.dim() == 3 else out.shape[0]
        rows.append([name, n, f"{1024 / n:.0f}x", f"{C.n_params(m) / 1e6:.1f}M"])
r.table(["connector", "tokens out", "compression", "parameters"], rows)

# ---------------------------------------------------------------- part 3: formula vs real checkpoints
r("")
r("Part 3. Connector parameters predicted from config.json, against the checkpoint headers (03_param_split).")
split_file = RESULTS / "03_param_split.json"
data = {"parity_max_abs_diff": worst, "shuffle_same_values": same_set, "tile": rows, "formula_vs_hub": []}
if not split_file.exists() or "--offline" in sys.argv:
    r("  skipped: run scripts/03_param_split.py first (needs network)")
else:
    from huggingface_hub import hf_hub_download

    measured = {m["repo"]: m for m in json.loads(split_file.read_text())["data"]["models"]}
    rows = []
    for repo, m in measured.items():
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        if cfg.get("model_type") not in C.FORMULA_TYPES:
            continue
        if "auto_map" not in cfg:  # AutoConfig fills in defaults that config.json leaves out (no remote code)
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(repo).to_dict()
        got = C.formula_params(cfg.get("model_type", ""), cfg)
        if got is None:
            continue
        pred, how = got
        hub = m["parts"]["connector"]
        rows.append([m["label"], f"{pred / 1e6:.2f}M", f"{hub / 1e6:.2f}M", "yes" if pred == hub else "no", how])
        data["formula_vs_hub"].append({"label": m["label"], "repo": repo, "formula": pred, "hub": hub, "how": how})
    r.table(["model", "formula", "checkpoint", "equal", "formula used"], rows)
r.save(data)
assert worst < 1e-5
