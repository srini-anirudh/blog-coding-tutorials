"""Section 11: the custom collator, and what goes wrong with a library default when pad == EOS.

Run: python c02-hf-stack/scripts/06_collator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from hfstack.chat import assistant_labels_by_prefix
from hfstack.collators import CausalLMCollator
from hfstack.data import make_arithmetic_rows, to_messages
from hfstack.utils import CHAT_LM, be_polite

be_polite()
tok = AutoTokenizer.from_pretrained(CHAT_LM)
print("pad token:", repr(tok.pad_token), tok.pad_token_id, "| eos token:", repr(tok.eos_token), tok.eos_token_id)

rows = make_arithmetic_rows(3, seed=1)
features = [assistant_labels_by_prefix(tok, to_messages(row)["messages"]) for row in rows]
print("example lengths:", [len(f["input_ids"]) for f in features])

collate = CausalLMCollator(pad_token_id=tok.pad_token_id)
batch = collate(features)
for key, value in batch.items():
    print(f"{key:<15} {tuple(value.shape)}")
print("\ntail of each row (last 8 positions)")
for i in range(batch["input_ids"].size(0)):
    print(f" row {i} input_ids", batch["input_ids"][i, -8:].tolist())
    print(f" row {i} attention", batch["attention_mask"][i, -8:].tolist())
    print(f" row {i} labels   ", batch["labels"][i, -8:].tolist())

print("\n--- break it: the library collator labels every token, then masks by VALUE ---")
library = DataCollatorForLanguageModeling(tok, mlm=False)
plain = [{"input_ids": f["input_ids"]} for f in features]
lib_batch = library(plain)
mine = CausalLMCollator(pad_token_id=tok.pad_token_id)(plain)
eos = tok.eos_token_id
real_eos = (mine["input_ids"] == eos) & (mine["attention_mask"] == 1)
print("real end-of-turn tokens in the batch :", int(real_eos.sum()))
print("of those, labelled by CausalLMCollator:", int((mine["labels"][real_eos] == eos).sum()))
print("of those, labelled by the library     :", int((lib_batch["labels"][real_eos] == eos).sum()))
