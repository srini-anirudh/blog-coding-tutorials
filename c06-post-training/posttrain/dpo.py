"""Stage 2: Direct Preference Optimization, written out by hand.

    loss = -log sigmoid( beta * [ (log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x)) ] )

Tokenization and batch layout follow TRL's DPOTrainer exactly ([chosen rows..., rejected rows...]), so
tests/test_dpo.py can feed the same batch to both and compare losses.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .chat import encode_prompt
from .generation import generate
from .logprobs import sequence_logprobs
from .tasks import Problem, exact_reward, near_miss, to_messages


def synthetic_pairs(problems: list[Problem], seed: int = 0) -> list[dict]:
    """Chosen = the correct worked solution. Rejected = the same solution with a carry-sized error."""
    rng = random.Random(seed)
    pairs = []
    for p in problems:
        pairs.append({
            "prompt": to_messages(p, with_answer=False),
            "chosen": [{"role": "assistant", "content": p.solution()}],
            "rejected": [{"role": "assistant", "content": near_miss(rng, p)}],
            "answer": p.answer,
        })
    return pairs


@torch.no_grad()
def on_policy_pairs(model, tok, problems: list[Problem], samples: int = 4, temperature: float = 1.0) -> list[dict]:
    """Sample from the SFT model; pair one of its correct answers with one of its wrong answers.

    Prompts where all samples are right (or all wrong) give no pair, which is itself a useful statistic.
    """
    pairs = []
    prompts = [encode_prompt(tok, to_messages(p, with_answer=False)) for p in problems]
    for s in range(0, len(problems), 16):
        chunk = problems[s : s + 16]
        gen = generate(model, tok, [q for q in prompts[s : s + 16] for _ in range(samples)], temperature=temperature)
        for i, p in enumerate(chunk):
            texts = gen.texts[i * samples : (i + 1) * samples]
            good = [t for t in texts if exact_reward(t, p.answer) == 1.0]
            bad = [t for t in texts if exact_reward(t, p.answer) == 0.0]
            if good and bad:
                pairs.append({
                    "prompt": to_messages(p, with_answer=False),
                    "chosen": [{"role": "assistant", "content": good[0]}],
                    "rejected": [{"role": "assistant", "content": bad[0]}],
                    "answer": p.answer,
                })
    return pairs


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def tokenize_pair(tok, pair: dict) -> dict[str, list[int]]:
    """prompt_ids, chosen_ids, rejected_ids, split the way TRL splits them.

    The prompt is rendered with the generation header; each completion is the rest of the full conversation,
    so it includes <|im_end|> and the template's trailing newline.
    """
    prompt_ids = encode_prompt(tok, pair["prompt"])
    full = {}
    for key in ("chosen", "rejected"):
        text = tok.apply_chat_template(pair["prompt"] + pair[key], tokenize=False)
        full[key] = tok(text, add_special_tokens=False)["input_ids"]
    n = min(_common_prefix(prompt_ids, full["chosen"]), _common_prefix(prompt_ids, full["rejected"]))
    return {"prompt_ids": prompt_ids[:n], "chosen_ids": full["chosen"][n:], "rejected_ids": full["rejected"][n:]}


def collate_pairs(examples: list[dict[str, list[int]]], pad_id: int, multiple: int = 1) -> dict[str, torch.Tensor]:
    """Right-padded (2B, L) batch: rows [0, B) are prompt+chosen, rows [B, 2B) are prompt+rejected.

    completion_mask is 1 on response tokens only, so the prompt never contributes to log pi(y|x).
    """
    rows, masks = [], []
    for key in ("chosen_ids", "rejected_ids"):
        for e in examples:
            rows.append(e["prompt_ids"] + e[key])
            masks.append([0] * len(e["prompt_ids"]) + [1] * len(e[key]))
    L = -(-max(len(r) for r in rows) // multiple) * multiple
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(rows), L), dtype=torch.long)
    cmask = torch.zeros((len(rows), L), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        ids[i, : len(r)] = torch.tensor(r)
        attn[i, : len(r)] = 1
        cmask[i, : len(m)] = torch.tensor(m)
    return {"input_ids": ids, "attention_mask": attn, "completion_mask": cmask}


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """All inputs are (B,) summed sequence log-probs. Returns the mean loss and the implicit rewards."""
    chosen_reward = beta * (policy_chosen - ref_chosen)  # implicit reward r = beta * log(pi / ref)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin).mean()
    return loss, {
        "chosen_reward": chosen_reward.detach(),
        "rejected_reward": rejected_reward.detach(),
        "margin": margin.detach(),
        "accuracy": (margin > 0).float().detach(),
    }


def dpo_batch_loss(policy, ref, batch: dict[str, torch.Tensor], beta: float = 0.1, average: bool = False):
    """Forward the policy (with grad) and the frozen reference (without) on one collated batch."""
    args = (batch["input_ids"], batch["attention_mask"], batch["completion_mask"])
    logps = sequence_logprobs(policy, *args, average=average)  # (2B,)
    with torch.no_grad():
        ref_logps = sequence_logprobs(ref, *args, average=average)
    pc, pr = logps.chunk(2)
    rc, rr = ref_logps.chunk(2)
    return dpo_loss(pc, pr, rc, rr, beta)
