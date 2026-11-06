"""After alignment: is the image really used, and what breaks the model silently?

    python c07-vlm/scripts/evaluate_alignment.py --run align-mlp-s2

Prints caption accuracy under five conditions and saves them to results/eval-<run>.json.
"""

from __future__ import annotations

import _common  # noqa: F401
import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from tinyvlm import build_tinyvlm, pick_device
from tinyvlm.data import COLORS, draw_shape, make_example, render, score

HERE = Path(__file__).resolve().parents[1]
QUESTION = "Describe the image."


def prompt_for(processor) -> str:
    message = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}
    return processor.apply_chat_template([message], add_generation_prompt=True)


@torch.no_grad()
def captions(model, processor, images, device, transform=None, padding_side="left", batch_size=16) -> list[str]:
    out = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        batch = processor([prompt_for(processor)] * len(chunk), [[im] for im in chunk], padding_side=padding_side)
        pixels = batch["pixel_values"].to(device)
        if transform is not None:
            pixels = transform(pixels)
        new = model.generate(
            batch["input_ids"].to(device), pixels, batch["attention_mask"].to(device),
            max_new_tokens=12, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id,
        )
        out += [p.strip() for p in processor.batch_decode(new)]
    return out


def out_of_distribution() -> dict[str, Image.Image]:
    """Pictures the projector never saw: the honest test of what it learned."""
    rng = random.Random(7)
    mixed = Image.new("RGB", (224, 224), (250, 250, 250))
    draw = ImageDraw.Draw(mixed)
    draw_shape(draw, "circle", 60, 60, 30, COLORS["red"])
    draw_shape(draw, "square", 160, 150, 30, COLORS["blue"])
    gradient = Image.linear_gradient("L").resize((224, 224)).convert("RGB")
    dark = render(2, "yellow", "triangle", rng)
    dark = Image.eval(dark, lambda v: 255 - v)  # inverted colours: black-ish canvas
    return {
        "a red circle and a blue square": mixed,
        "four green squares (count outside 1-3)": _four(rng),
        "two inverted triangles on a dark canvas": dark,
        "a grey gradient, no shapes": gradient,
    }


def _four(rng: random.Random) -> Image.Image:
    image = Image.new("RGB", (224, 224), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    for cx, cy in ((55, 55), (165, 55), (55, 165), (165, 165)):
        draw_shape(draw, "square", cx, cy, rng.randint(20, 30), COLORS["green"])
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="align-mlp-s2")
    parser.add_argument("--projector", default="mlp")
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    model, processor = build_tinyvlm(projector_type=args.projector)
    model.projector.load_state_dict(load_file(HERE / "runs" / args.run / "projector.safetensors"))
    model.to(device).eval()

    examples = [make_example(i, seed=1) for i in range(args.n)]
    images = [e["images"][0] for e in examples]
    targets = [e["target"] for e in examples]
    report: dict = {"run": args.run, "device": device.type, "n": args.n, "conditions": {}}

    conditions = {
        "correct": {},
        "blind (all pixels zero)": {"transform": torch.zeros_like},
        "wrong normalisation ([0, 1], not [-1, 1])": {"transform": lambda x: x * 0.5 + 0.5},
        "right padding at generation": {"padding_side": "right"},
    }
    for name, kwargs in conditions.items():
        predicted = captions(model, processor, images, device, **kwargs)
        report["conditions"][name] = {"score": score(predicted, targets), "samples": predicted[:4]}
        print(f"{name:<44} {report['conditions'][name]['score']}")

    # Silent bug: pixel_values reversed against the placeholders. Nothing raises; captions swap.
    shifted = captions(model, processor, images[:16], device, transform=lambda x: x.flip(0))
    report["conditions"]["crops reversed inside the batch"] = {
        "score": score(shifted, targets[:16]), "score_against_reversed_targets": score(shifted, targets[:16][::-1]),
    }
    print("crops reversed inside the batch              ", report["conditions"]["crops reversed inside the batch"])

    # Did training also fix the scale? Compare the trained projector's output with real token embeddings.
    with torch.no_grad():
        batch = processor([prompt_for(processor)], [images[0]])
        features = model.get_image_features(batch["pixel_values"].to(device))
        embeddings = model.get_input_embeddings().weight[: model.config.image_token_id]
    report["scale"] = {"trained_projector_std": features.std().item(), "token_embedding_std": embeddings.std().item()}
    print("scale", report["scale"])

    report["samples"] = [
        {"target": e["messages"][1]["content"], "predicted": p}
        for e, p in zip(examples[:8], report["conditions"]["correct"]["samples"] + captions(model, processor, images[4:8], device))
    ]
    report["out_of_distribution"] = {}
    for name, image in out_of_distribution().items():
        report["out_of_distribution"][name] = captions(model, processor, [image], device)[0]
        print(f"{name:<44} -> {report['out_of_distribution'][name]!r}")
    (HERE / "results" / f"eval-{args.run}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
