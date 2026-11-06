"""Image-conditioned DPO, written out by hand.

    loss = -log sigmoid( beta * [ (log pi(y_w|x,I) - log ref(y_w|x,I))
                                 - (log pi(y_l|x,I) - log ref(y_l|x,I)) ] )

The image I sits inside every term. That is the only change from text DPO, and
it is why the blind-preference check (blind.py) matters: nothing in this loss
forces the model to use I.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .logprobs import SeqBatch, build_batch, sequence_logprobs
from .model import encode_answer, encode_prompt
from .preference import PreferencePair


def dpo_loss(pi_chosen: torch.Tensor, pi_rejected: torch.Tensor, ref_chosen: torch.Tensor,
             ref_rejected: torch.Tensor, beta: float = 0.1) -> dict[str, torch.Tensor]:
    """All inputs (P,) sequence log-probs for P pairs. Returns the loss and diagnostics."""
    chosen_reward = beta * (pi_chosen - ref_chosen)        # implicit reward of y_w
    rejected_reward = beta * (pi_rejected - ref_rejected)  # implicit reward of y_l
    margin = chosen_reward - rejected_reward
    return {
        "loss": -F.logsigmoid(margin).mean(),
        "margin": margin.detach(),
        "chosen_reward": chosen_reward.detach(),
        "rejected_reward": rejected_reward.detach(),
        "accuracy": (margin > 0).float().mean().detach(),
    }


def pair_batch(processor, pairs: list[PreferencePair], images=None) -> SeqBatch:
    """2P rows: row 2i is pair i's chosen answer, row 2i+1 its rejected answer.

    Both rows of a pair point at the same image (image_index i), so the vision
    encoder runs P times, not 2P. Pass `images` to swap in other images
    (the blind check passes grey images here).
    """
    prompts, answers, image_index, pixels = [], [], [], []
    for i, pair in enumerate(pairs):
        img = pair.image() if images is None else images[i]
        p = encode_prompt(processor, img, pair.prompt)
        pixels.append(p.pixel_values)
        for answer in (pair.chosen, pair.rejected):
            prompts.append(p.input_ids[0])
            answers.append(encode_answer(processor, answer))
            image_index.append(i)
    return build_batch(prompts, answers, image_index, torch.cat(pixels), processor.tokenizer.pad_token_id)


def pair_logprobs(model, batch: SeqBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """(chosen, rejected) sequence log-probs, each (P,)."""
    logp, _ = sequence_logprobs(model, batch)
    return logp[0::2], logp[1::2]


@torch.no_grad()
def reference_logprobs(model, processor, pairs: list[PreferencePair], batch_pairs: int = 8,
                       device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Score every pair once with the frozen starting model.

    The reference never changes, so precomputing it removes the second model
    from memory and halves the forward passes during training.
    """
    model.eval()
    chosen, rejected = [], []
    for s in range(0, len(pairs), batch_pairs):
        c, r = pair_logprobs(model, pair_batch(processor, pairs[s: s + batch_pairs]).to(device))
        chosen.append(c.cpu())
        rejected.append(r.cpu())
    return torch.cat(chosen), torch.cat(rejected)
