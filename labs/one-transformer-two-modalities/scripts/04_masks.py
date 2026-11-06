"""Section 5: causal vs bidirectional. One boolean, large consequences.

Same weights, two masks. With the causal mask, changing the future never changes the past, a KV cache is exact,
and one pass gives N - 1 training targets. Without it, every output depends on every input, and a cache is wrong.
Then the VLM's block mask: a bidirectional image prefix, a causal text suffix. Runs offline.
"""

import _setup
import torch

from twomod import BlockConfig, TextTransformer
from twomod.masks import bidirectional, causal, prefix_lm, render

_setup.header()
torch.manual_seed(0)

_setup.section("The two masks")
print("causal (language model)        bidirectional (ViT)")
for a, b in zip(render(causal(6)).split("\n"), render(bidirectional(6)).split("\n")):
    print(f"{a:<31}{b}")

cfg = BlockConfig(dim=64, n_heads=4)
lm = TextTransformer(500, cfg, depth=3, causal=True).eval()
enc = TextTransformer(500, cfg, depth=3, causal=False).eval()
enc.load_state_dict(lm.state_dict())  # identical weights: only the mask differs

_setup.section("Does changing the future change the past?")
ids = torch.randint(0, 500, (1, 32))
changed = ids.clone()
changed[:, 20:] = torch.randint(0, 500, (1, 12))  # rewrite tokens 20..31
with torch.no_grad():
    for name, model in [("causal", lm), ("bidirectional", enc)]:
        d = (model(ids)[:, :20] - model(changed)[:, :20]).abs().max().item()
        print(f"{name:<14} logits at positions 0..19 after rewriting 20..31, max |difference|: {d:.1e}")

_setup.section("A KV cache is exact only under the causal mask")
prompt, steps = ids[:, :8], ids[:, 8:24]
with torch.no_grad():
    for name, model in [("causal", lm), ("bidirectional", enc)]:
        caches = [{} for _ in model.blocks]
        outs = [model(prompt, caches=caches, start=0)[:, -1]]
        for t in range(steps.shape[1]):  # feed one token at a time; old K and V are reused, never recomputed
            outs.append(model(steps[:, t : t + 1], caches=caches, start=8 + t)[:, -1])
        cached = torch.stack(outs[:-1], dim=1)
        full = model(ids[:, :24])[:, 7:23]
        print(f"{name:<14} cached decoding vs a full forward pass, max |logit difference| over 16 steps: "
              f"{(cached - full).abs().max().item():.1e}")
print("bidirectional: when token t+1 arrives, every earlier key and value would change, so there is nothing to keep")

_setup.section("Supervision per forward pass")
n = 256
print(f"a text sequence of {n} tokens: {n - 1} next-token targets from one pass (loss over every position)")
print("an image of 196 patches with a class label: 1 target from one pass")
loss = lm.loss(torch.randint(0, 500, (2, n)))
print(f"causal LM loss at init on random tokens: {loss.item():.3f} (ln 500 = {torch.log(torch.tensor(500.)):.3f})")

_setup.section("A VLM uses both: an image prefix that is bidirectional, text that is causal")
print("4 image tokens, then 5 text tokens:")
print(render(prefix_lm(9, 4)))
