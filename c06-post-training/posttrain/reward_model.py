"""Stage 3 (optional): a pairwise reward model.

A transformer body plus one linear head that reads the hidden state of the last real token and outputs a
scalar score. Trained with the Bradley-Terry loss  -log sigmoid(r(x, y_w) - r(x, y_l)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class RewardModel(nn.Module):
    def __init__(self, name_or_path: str):
        super().__init__()
        self.body = AutoModel.from_pretrained(name_or_path)  # the decoder without its LM head
        self.head = nn.Linear(self.body.config.hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, L) right-padded -> (B,) scores, read at each row's last real token."""
        h = self.body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        last = attention_mask.sum(-1) - 1  # (B,) index of the last non-pad token
        h_last = h[torch.arange(h.shape[0], device=h.device), last]  # (B, D)
        return self.head(h_last).squeeze(-1)


def pairwise_loss(chosen: torch.Tensor, rejected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bradley-Terry: (B,), (B,) -> mean loss, accuracy (fraction with chosen scored higher)."""
    return -F.logsigmoid(chosen - rejected).mean(), (chosen > rejected).float().mean()
