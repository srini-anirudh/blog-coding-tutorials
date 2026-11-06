"""Two methods with one name.

Contrastive decoding (Li et al., 2022): score = log p_expert - beta * log p_amateur, restricted to the plausible set
{ i : p_expert(i) >= alpha * max_j p_expert(j) }. A second model, no hidden states.

Contrastive search (Su et al., 2022): re-rank the model's own top-k by (1 - a) * p(v) - a * max_j cos(h_v, h_j), where
h_j are the hidden states of the context. One model, repetition measured in representation space.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import DynamicCache


def contrastive_scores(expert_logits: torch.Tensor, amateur_logits: torch.Tensor, alpha: float | None = 0.1,
                       beta: float = 1.0) -> torch.Tensor:
    le = torch.log_softmax(expert_logits.float(), -1)
    la = torch.log_softmax(amateur_logits.float(), -1)
    score = le - beta * la
    if alpha is not None:
        plausible = le >= torch.log(torch.tensor(alpha)) + le.max(-1, keepdim=True).values
        score = score.masked_fill(~plausible, float("-inf"))
    return score


@torch.no_grad()
def contrastive_decode(expert, amateur, prompt_ids: torch.Tensor, max_new_tokens: int, alpha: float | None = 0.1,
                       beta: float = 1.0) -> torch.Tensor:
    """Greedy over the contrastive score. Both models keep their own KV cache. Returns (1, K)."""
    ce, ca = DynamicCache(config=expert.config), DynamicCache(config=amateur.config)
    step, out = prompt_ids, []
    for _ in range(max_new_tokens):
        le = expert(input_ids=step, past_key_values=ce, use_cache=True).logits[:, -1]
        la = amateur(input_ids=step, past_key_values=ca, use_cache=True).logits[:, -1]
        nxt = contrastive_scores(le, la, alpha, beta).argmax(-1)
        out.append(nxt)
        step = nxt[:, None]
    return torch.stack(out, 1)


@torch.no_grad()
def contrastive_search(model, prompt_ids: torch.Tensor, max_new_tokens: int, k: int = 5,
                       penalty: float = 0.6) -> torch.Tensor:
    """Su et al.'s contrastive search for one prompt (1, T). One forward of k candidate tokens per step."""
    cache = DynamicCache(config=model.config)
    out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
    context_h = F.normalize(out.hidden_states[-1][0].float(), dim=-1)  # (T, d)
    logits = out.logits[:, -1].float()
    chosen = []
    for _ in range(max_new_tokens):
        probs = torch.softmax(logits, -1)[0]
        top_p, top_i = torch.topk(probs, k)
        cache.reorder_cache(torch.zeros(k, dtype=torch.long))  # copy the context cache to k rows
        o = model(input_ids=top_i[:, None], past_key_values=cache, use_cache=True, output_hidden_states=True)
        cand_h = F.normalize(o.hidden_states[-1][:, -1].float(), dim=-1)  # (k, d)
        degeneration = (cand_h @ context_h.T).max(-1).values
        best = int(torch.argmax((1 - penalty) * top_p - penalty * degeneration))
        cache.reorder_cache(torch.tensor([best]))
        chosen.append(int(top_i[best]))
        context_h = torch.cat([context_h, cand_h[best:best + 1]], 0)
        logits = o.logits[best:best + 1, -1].float()
    return torch.tensor([chosen])
