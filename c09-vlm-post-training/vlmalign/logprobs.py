"""Batches of (prompt, answer) sequences and their log-probabilities under the VLM.

Everything in DPO and GRPO reduces to one quantity: log pi(answer | image, prompt),
the sum of per-token log-probabilities over the answer tokens only.

Shapes used below
    B   sequences in the batch (2 per preference pair: chosen and rejected)
    T   padded sequence length (prompt + answer)
    N   image tokens per image (64 for SmolVLM at 512x512 without tiling)
    D   decoder hidden size (576)
    V   vocabulary size (49,280)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

IGNORE = -100


@dataclass
class SeqBatch:
    input_ids: torch.Tensor       # (B, T)   prompt ids, then answer ids, then pad
    attention_mask: torch.Tensor  # (B, T)   1 on real tokens, 0 on right padding
    labels: torch.Tensor          # (B, T)   answer ids; IGNORE on prompt and pad
    image_index: torch.Tensor     # (B,)     which image (row of `pixel_values`) each sequence looks at
    pixel_values: torch.Tensor    # (I, 1, 3, H, W)  one entry per DISTINCT image

    def to(self, device) -> "SeqBatch":
        return SeqBatch(*(t.to(device) for t in (self.input_ids, self.attention_mask, self.labels,
                                                  self.image_index, self.pixel_values)))


def build_batch(prompts: list[torch.Tensor], answers: list[torch.Tensor], image_index: list[int],
                pixel_values: torch.Tensor, pad_id: int) -> SeqBatch:
    """Right-pad prompt+answer rows. prompts[i]: (T_p,), answers[i]: (T_a,)."""
    rows = [torch.cat([p, a]) for p, a in zip(prompts, answers)]
    T = max(len(r) for r in rows)
    B = len(rows)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    labels = torch.full((B, T), IGNORE, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompts, rows)):
        input_ids[i, : len(r)] = r
        attention_mask[i, : len(r)] = 1
        labels[i, len(p): len(r)] = r[len(p):]  # loss only on the answer
    return SeqBatch(input_ids, attention_mask, labels, torch.tensor(image_index), pixel_values)


def image_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Vision encoder + connector. (I, 1, 3, H, W) -> (I, N, D)."""
    out = model.model.get_image_features(pixel_values, return_dict=True)
    return torch.stack(list(out.pooler_output)) if isinstance(out.pooler_output, (list, tuple)) else out.pooler_output


def merge_image_tokens(model, input_ids: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    """Write image features into the <image> placeholder slots.

    input_ids: (B, T), feats: (B, N, D) -> inputs_embeds: (B, T, D).
    The b-th row's N placeholders receive feats[b] in order.
    """
    embeds = model.get_input_embeddings()(input_ids)                      # (B, T, D)
    slots = (input_ids == model.config.image_token_id).unsqueeze(-1)      # (B, T, 1)
    return embeds.masked_scatter(slots, feats.to(embeds.dtype).reshape(-1, feats.shape[-1]))


def forward_logits(model, batch: SeqBatch, encode_once: bool = True) -> torch.Tensor:
    """Logits (B, T, V).

    encode_once=False is the naive path: hand pixel_values to the model, which
    runs the vision encoder once per ROW. Chosen and rejected share an image, and
    G rollouts share an image, so that repeats the same encoder work 2x or Gx.
    encode_once=True runs the encoder once per DISTINCT image and indexes into it.
    Gradients still flow into the vision tower and the connector either way.
    """
    if not encode_once:
        pv = batch.pixel_values[batch.image_index]                        # (B, 1, 3, H, W)
        return model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, pixel_values=pv).logits
    feats = image_features(model, batch.pixel_values)                     # (I, N, D)
    embeds = merge_image_tokens(model, batch.input_ids, feats[batch.image_index])
    return model(inputs_embeds=embeds, attention_mask=batch.attention_mask).logits


def token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log p(label) and its mask, shifted so position t predicts token t+1.

    logits (B, T, V), labels (B, T) -> logp (B, T-1), mask (B, T-1) bool.
    """
    logits = logits[:, :-1].float()          # log-softmax in fp32: bf16 here loses ~0.1 nats
    targets = labels[:, 1:]
    mask = targets != IGNORE
    logp = torch.gather(F.log_softmax(logits, dim=-1), 2, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return logp * mask, mask


def sequence_logprobs(model, batch: SeqBatch, encode_once: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of answer-token log-probs per row, and the answer length. Both (B,)."""
    logp, mask = token_logprobs(forward_logits(model, batch, encode_once), batch.labels)
    return logp.sum(-1), mask.sum(-1)
