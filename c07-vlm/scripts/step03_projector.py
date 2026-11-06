"""Step 3: the projector, Linear then MLP, and pixel shuffle. Compare scales with real token embeddings."""

import _common  # noqa: F401
import random

import torch

from _common import stats
from tinyvlm import build_tinyvlm
from tinyvlm.data import render
from tinyvlm.projector import LinearProjector, MLPProjector, pixel_shuffle

model, processor = build_tinyvlm(projector_type="mlp", pixel_shuffle_factor=2)
pixels = processor.image_processor([render(2, "red", "circle", random.Random(0))])["pixel_values"]
with torch.no_grad():
    features = model.vision(pixel_values=pixels).last_hidden_state
    shuffled = pixel_shuffle(features, 2)
    linear, mlp = LinearProjector(768, 576), MLPProjector(768, 576)
    print(stats("vision features", features))
    print(stats("after pixel shuffle", shuffled))
    print(stats("Linear(768)", linear(features)))
    print(stats("MLP(768)", mlp(features)))
    print(stats("MLP(3072), the model", model.projector(shuffled)))
    ids = processor.tokenizer("A photo of two red circles on a white table.", return_tensors="pt")["input_ids"]
    print(stats("real token embeddings", model.get_input_embeddings()(ids)))
    print(stats("all embedding rows", model.get_input_embeddings().weight[:49152]))
print("parameters")
for name, module in (("Linear 768 -> 576", linear), ("MLP 768 -> 576", mlp), ("Linear 3072 -> 576", LinearProjector(3072, 576)), ("MLP 3072 -> 576", model.projector)):
    print(f"  {name:<20} {sum(p.numel() for p in module.parameters()):>10,}")
print("whole model", model.parameter_counts())
