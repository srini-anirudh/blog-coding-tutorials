"""Visual GRPO: G rollouts per (image, prompt), group-relative advantages, policy update.

    image + prompt --> G sampled answers --> reward r_1..r_G (verifiable, from the scene)
                   --> A_i = (r_i - mean(r)) / (std(r) + eps)
                   --> push up log pi of tokens in above-average answers, down for below

Shapes
    P   prompts (images) per step
    G   rollouts per prompt
    L   padded completion length
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .logprobs import SeqBatch, build_batch, forward_logits, token_logprobs
from .model import PromptInputs, end_of_utterance_id


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4, scale_by_std: bool = True) -> torch.Tensor:
    """rewards (P, G) -> advantages (P, G).

    Each answer is compared only with the other answers to the SAME image. A
    group where all G answers score the same has zero advantage everywhere: it
    teaches nothing, however good or bad the answers are.
    """
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if not scale_by_std:  # Dr. GRPO drops the std to avoid up-weighting near-uniform groups
        return centered
    return centered / (rewards.std(dim=1, keepdim=True) + eps)


@dataclass
class Rollouts:
    prompt_ids: torch.Tensor      # (T_p,)
    pixel_values: torch.Tensor    # (1, 1, 3, H, W)
    completions: torch.Tensor     # (G, L)  sampled ids, end_of_utterance kept, pad after it
    lengths: torch.Tensor         # (G,)    tokens up to and including end_of_utterance
    texts: list[str]


@torch.no_grad()
def sample_group(model, processor, prompt: PromptInputs, group_size: int, max_new_tokens: int = 16,
                 temperature: float = 1.0, top_p: float = 1.0) -> Rollouts:
    """G sampled answers for one image. temperature=1.0, top_p=1.0 samples the policy itself."""
    eou, pad = end_of_utterance_id(processor), processor.tokenizer.pad_token_id
    p = prompt.to(model.device)
    out = model.generate(input_ids=p.input_ids, attention_mask=torch.ones_like(p.input_ids),
                         pixel_values=p.pixel_values, do_sample=True, temperature=temperature, top_p=top_p,
                         top_k=0, num_return_sequences=group_size, max_new_tokens=max_new_tokens,
                         eos_token_id=eou, pad_token_id=pad)
    comp = out[:, p.input_ids.shape[1]:].cpu()                       # (G, L)
    is_eou = comp == eou
    # length = first end_of_utterance (inclusive), or L if the model never stopped
    first = torch.where(is_eou.any(1), is_eou.float().argmax(1) + 1, torch.full((len(comp),), comp.shape[1]))
    comp = comp.masked_fill(torch.arange(comp.shape[1]) >= first[:, None], pad)
    texts = processor.batch_decode(comp, skip_special_tokens=True)
    return Rollouts(p.input_ids[0].cpu(), p.pixel_values.cpu(), comp, first, texts)


def rollout_batch(groups: list[Rollouts], pad_id: int) -> SeqBatch:
    """Stack P groups of G completions into one batch; rows of a group share one image."""
    prompts, answers, image_index = [], [], []
    for i, g in enumerate(groups):
        for comp, n in zip(g.completions, g.lengths):
            prompts.append(g.prompt_ids)
            answers.append(comp[: int(n)])
            image_index.append(i)
    return build_batch(prompts, answers, image_index, torch.cat([g.pixel_values for g in groups]), pad_id)


def grpo_loss(logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
              clip_eps: float = 0.2, ref_logp: torch.Tensor | None = None, beta: float = 0.0) -> torch.Tensor:
    """Clipped policy-gradient loss averaged over all completion tokens.

    logp, old_logp, ref_logp, mask: (B, T-1) per token. adv: (B,), one value per rollout,
    broadcast to every token of that rollout. With one update per batch,
    old_logp == logp.detach(), the ratio is 1, and the gradient is A * grad log pi.
    """
    ratio = torch.exp(logp - old_logp)
    a = adv[:, None]
    per_token = -torch.min(ratio * a, ratio.clamp(1 - clip_eps, 1 + clip_eps) * a)
    if ref_logp is not None and beta > 0:
        # k3 estimator of KL(pi || ref): unbiased, always >= 0
        d = ref_logp - logp
        per_token = per_token + beta * (torch.exp(d) - d - 1)
    return (per_token * mask).sum() / mask.sum().clamp_min(1)


def policy_token_logprobs(model, batch: SeqBatch) -> tuple[torch.Tensor, torch.Tensor]:
    return token_logprobs(forward_logits(model, batch), batch.labels)
