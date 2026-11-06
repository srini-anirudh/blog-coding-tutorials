"""Section 6: messages -> text -> token IDs -> assistant-only labels.

Run: python c02-hf-stack/scripts/03_chat_template.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from hfstack.chat import assistant_labels_by_prefix, assistant_labels_by_template
from hfstack.utils import CHAT_LM, be_polite

be_polite()
tok = AutoTokenizer.from_pretrained(CHAT_LM)
messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]

print("--- the template renders plain text ---")
print(repr(tok.apply_chat_template(messages, tokenize=False)))

print("\n--- add_generation_prompt stops right where the model should start writing ---")
print(repr(tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)))

print("\n--- tokenize=True returns a BatchEncoding in transformers 5 ---")
out = tok.apply_chat_template(messages, tokenize=True)
print(type(out).__name__, list(out.keys()))

print("\n--- token by token, with the assistant-only labels ---")
example = assistant_labels_by_prefix(tok, messages)
print(f"{'pos':>3} {'id':>6} {'token':<14} {'label':>6}")
for pos, (i, label) in enumerate(zip(example["input_ids"], example["labels"])):
    print(f"{pos:>3} {i:>6} {tok.convert_ids_to_tokens(i)!r:<14} {label:>6}")

other = assistant_labels_by_template(tok, messages)
print("\nprefix diffing == generation tags:", example["labels"] == other["labels"])

print("\n--- a template adds what you did not ask for: no system message given ---")
print(repr(tok.apply_chat_template(messages[1:], tokenize=False)))
