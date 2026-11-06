"""Section 3: assembling the sequence.

Text is ragged: special tokens, a chat grammar, padding (and the waste it costs), or packing.
Images are rectangular: N = (H/P)(W/P) every time, plus an optional [CLS] and optional registers.

Needs wikitext-2, the GPT-2 tokenizer and the SmolLM2-135M-Instruct tokenizer from the Hugging Face Hub.
"""

import random

import _setup
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from twomod import BlockConfig, TextTransformer, ViT
from twomod.masks import causal, render, with_padding

_setup.header()
torch.manual_seed(0)

_setup.section("Special tokens and the chat grammar")
chat_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
messages = [{"role": "user", "content": "What is a patch?"}, {"role": "assistant", "content": "A tile of pixels."}]
rendered = chat_tok.apply_chat_template(messages, tokenize=False)
content = sum(len(chat_tok.encode(m["content"])) for m in messages)
total = len(chat_tok.encode(rendered))
print(f"SmolLM2-135M-Instruct chat template:\n{rendered.rstrip()}")
print(f"content tokens {content}, tokens after the template {total}: {total - content} are structure")

_setup.section("Padding: the cost of a ragged batch")
lengths = [40, 900]
longest = max(lengths)
print(f"lengths {lengths} padded to {longest}: the short row is {(longest - 40) / longest:.1%} padding; "
      f"the whole batch is {sum(longest - n for n in lengths) / (longest * len(lengths)):.1%} padding")

gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]["text"]
docs, cur = [], []
for line in wiki:  # articles start with a " = Title = " line
    if line.startswith(" = ") and not line.startswith(" = = ") and cur:
        docs.append("".join(cur))
        cur = []
    cur.append(line)
docs.append("".join(cur))
paras = [p for d in docs for p in d.split("\n") if p.strip() and not p.startswith(" =")]
para_lens = [len(gpt2.encode(p)) for p in paras]
rng = random.Random(0)
for name, lens in [("articles", [len(gpt2.encode(d)) for d in docs]), ("paragraphs", para_lens)]:
    waste = []
    for _ in range(200):
        batch = [min(n, 1024) for n in rng.sample(lens, 16)]  # truncate to a 1,024-token context
        waste.append(1 - sum(batch) / (16 * max(batch)))
    print(f"wikitext-2 {name} ({len(lens):,}, median {sorted(lens)[len(lens) // 2]} tokens), random batches of 16, "
          f"cut at 1,024: mean padding {sum(waste) / len(waste):.1%} of the batch")
print("packing the same tokens end to end into full rows leaves only the last row partly empty")

print("\nPadding is correct only if it is masked. A causal mask combined with a padding mask:")
keep = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
print(render(with_padding(causal(6), keep)[1]))
cfg = BlockConfig(dim=64, n_heads=4)
model = TextTransformer(1000, cfg, depth=2).eval()
short = torch.randint(0, 1000, (1, 3))
padded = torch.cat([short, torch.zeros(1, 3, dtype=torch.long)], dim=1)  # right padding with id 0
with torch.no_grad():
    alone, in_batch = model(short), model(padded)[:, :3]
print(f"right-padded row vs the same row alone, max |logit difference| on real tokens: "
      f"{(alone - in_batch).abs().max().item():.1e} (a causal mask already hides PAD keys on the right)")

_setup.section("Image sequences: fixed length, set by geometry")
print(f"{'resolution':>10} {'patch':>5} {'N tokens':>9} {'N vs 224':>9} {'scores vs 224':>14}")
for patch in (16, 14):
    base = (224 // patch) ** 2
    for res in (224, 336, 448, 896):
        n = (res // patch) ** 2
        print(f"{res:>10} {patch:>5} {n:>9,} {n / base:>8.2f}x {(n / base) ** 2:>13.2f}x")

vit = ViT(BlockConfig(dim=64, n_heads=4), depth=2, image_size=224, patch=16, num_classes=10, registers=4)
images = torch.randn(3, 3, 224, 224)
print(f"\nViT with [CLS] and 4 registers: tokens {tuple(vit.tokens(images).shape)} = 1 + 4 + 196 for every image; "
      f"no padding, no mask; logits {tuple(vit(images).shape)}")
