"""Language-model plumbing, borrowed from C6 rather than rebuilt.

C6 (c06-post-training/posttrain) already writes out sequence log-probabilities, the chosen/rejected batch layout
and a tiny random Llama. This module puts that package on the path and adds two small helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

C06 = Path(__file__).resolve().parents[3] / "c06-post-training"
if str(C06) not in sys.path:
    sys.path.insert(0, str(C06))

from posttrain.chat import load_tokenizer  # noqa: E402,F401
from posttrain.dpo import collate_pairs, tokenize_pair  # noqa: E402,F401
from posttrain.dpo import dpo_loss as c06_dpo_loss  # noqa: E402,F401
from posttrain.logprobs import sequence_logprobs  # noqa: E402
from posttrain.models import frozen_copy, tiny_llama  # noqa: E402,F401

INSTRUCT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def pair_logprobs(model, batch: dict[str, torch.Tensor], average: bool = False):
    """Summed (or mean) log pi(y | x) for the chosen rows and the rejected rows of a C6-layout batch."""
    lp = sequence_logprobs(model, batch["input_ids"], batch["attention_mask"], batch["completion_mask"],
                           average=average)
    return lp.chunk(2)


def toy_pair(vocab: int = 128, seed: int = 0) -> dict[str, torch.Tensor]:
    """One hand-made token pair for the tiny Llama: a shared prompt, then two different completions."""
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(3, vocab, (6,), generator=g).tolist()
    chosen = torch.randint(3, vocab, (5,), generator=g).tolist() + [2]
    rejected = torch.randint(3, vocab, (7,), generator=g).tolist() + [2]
    return collate_pairs([{"prompt_ids": prompt, "chosen_ids": chosen, "rejected_ids": rejected}], pad_id=0)
