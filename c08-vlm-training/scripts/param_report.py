"""Trainable parameters per module for every freeze preset and LoRA placement.

    python c08-vlm-training/scripts/param_report.py

Writes results/param_report.json, which the configurator figure in the post reads its numbers from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmtrain.c07_adapter import build_c07  # noqa: E402
from vlmtrain.freeze import format_report, set_trainable, trainable_report, vision_layers  # noqa: E402
from vlmtrain.lora import apply_lora  # noqa: E402

VARIANTS = [
    ("align", {"preset": "align"}), ("lm", {"preset": "lm"}),
    ("partial top 4", {"preset": "partial", "k": 4}), ("full", {"preset": "full"}),
    ("lora lm", {"preset": "none", "lora": ["lm"]}),
    ("lora vision", {"preset": "none", "lora": ["vision"]}),
    ("lora projector", {"preset": "none", "lora": ["projector"]}),
    ("lora all", {"preset": "none", "lora": ["lm", "vision", "projector"]}),
]


def main() -> None:
    out: dict = {"variants": {}}
    for name, v in VARIANTS:
        _, model, _ = build_c07()
        set_trainable(model, v["preset"], v.get("k", 0))
        wrapped = apply_lora(model, v["lora"], r=16, alpha=32) if "lora" in v else []
        rep = trainable_report(model)
        out["variants"][name] = {"report": rep, "lora_linears": len(wrapped)}
        print(f"--- {name}" + (f" ({len(wrapped)} wrapped linears, r=16)" if wrapped else ""))
        print(format_report(rep))
        if name == "align":
            out["vision_layer_params"] = [sum(p.numel() for p in layer.parameters()) for layer in vision_layers(model)]
            out["lm_layer_params"] = [sum(p.numel() for p in layer.parameters()) for layer in model.lm.model.layers]
            out["lm_embed_params"] = model.lm.get_input_embeddings().weight.numel()
            out["vision_total"] = rep["vision"]["total"]
            out["lm_total"] = rep["lm"]["total"]
            out["projector_total"] = rep["projector"]["total"]
    path = Path(__file__).resolve().parent.parent / "results" / "param_report.json"
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
