"""Step 5 and 6: the multimodal sequence, printed token by token, with labels."""

import _common  # noqa: F401
import random

from tinyvlm import IGNORE_INDEX, build_tinyvlm
from tinyvlm.data import render

_, processor = build_tinyvlm()
image = render(2, "red", "circle", random.Random(0))
messages = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see?"}]},
    {"role": "assistant", "content": "Two red circles."},
]
print(repr(processor.apply_chat_template(messages)))
example = processor.build_example(messages, [image])
tokens = processor.tokenizer.convert_ids_to_tokens(example["input_ids"])
print(f"{'pos':>4} {'id':>6} {'label':>6}  token")
shown_image = False
for t, (token, i, label) in enumerate(zip(tokens, example["input_ids"], example["labels"])):
    if i == processor.image_token_id:
        if not shown_image:
            print(f"{t:>4} {i:>6} {label:>6}  <image> x {example['input_ids'].count(i)}")
            shown_image = True
        continue
    print(f"{t:>4} {i:>6} {label:>6}  {token!r}")
n = len(example["labels"])
print("sequence length", n, "| supervised", sum(lab != IGNORE_INDEX for lab in example["labels"]), "| ignored", sum(lab == IGNORE_INDEX for lab in example["labels"]))
