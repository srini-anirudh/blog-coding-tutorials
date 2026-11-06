"""Did RL change what the image features contain?

Fit a linear probe from mean-pooled image features to the number of each shape.
Compare the pretrained features with the features of a trained checkpoint. The
same probe on the same scenes: only the features differ.

    python scripts/perception_probe.py --runs grpo_count_exact_all grpo_count_exact_projector_llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlmalign.device import pick_device, set_threads  # noqa: E402
from vlmalign.freeze import configure_trainable  # noqa: E402
from vlmalign.logprobs import image_features  # noqa: E402
from vlmalign.model import encode_prompt, load_model, load_processor  # noqa: E402
from vlmalign.scenes import SHAPES, render, sample_scene  # noqa: E402


@torch.no_grad()
def features(model, proc, scenes) -> torch.Tensor:
    """(S, D): mean over the 64 image tokens of each scene."""
    out = []
    for s in scenes:
        p = encode_prompt(proc, render(s), "x").to(model.device)
        out.append(image_features(model, p.pixel_values).mean(1).cpu())
    return torch.cat(out)


def probe_accuracy(xtr, ytr, xte, yte) -> float:
    """Ridge regression to the count of each shape; one hit per shape whose count rounds to the truth."""
    mu, sd, ym = xtr.mean(0), xtr.std(0) + 1e-6, ytr.mean(0)  # ym is the intercept: centred features have none
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    w = torch.linalg.solve(xtr.T @ xtr + 10.0 * torch.eye(xtr.shape[1]), xtr.T @ (ytr - ym))
    return float((((xte @ w) + ym).round() == yte).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--test", type=int, default=200)
    args = ap.parse_args()
    set_threads()
    device = pick_device()
    proc = load_processor()
    scenes = [sample_scene(900_000 + i, with_word=False) for i in range(args.train + args.test)]
    y = torch.tensor([[s.count(k) for k in SHAPES] for s in scenes], dtype=torch.float32)
    ytr, yte = y[: args.train], y[args.train:]

    results = {}
    print(f"{'features':36s} {'per-shape count accuracy':>26s}")
    for run in ["pretrained"] + args.runs:
        model = load_model(device=device).eval()
        if run != "pretrained":
            saved = json.loads((ROOT / "results" / f"{run}.json").read_text())
            configure_trainable(model, saved["args"]["freeze"])
            model.load_state_dict(torch.load(ROOT / "runs" / f"{run}.pt", map_location=device), strict=False)
        x = features(model, proc, scenes)
        acc = probe_accuracy(x[: args.train], ytr, x[args.train:], yte)
        print(f"{run:36s} {acc:26.3f}")
        results[run] = acc
        del model
    (ROOT / "results" / "perception_probe.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
