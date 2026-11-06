"""Collate trajectories into padded tensors for one forward pass over whole episodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .trajectory import Trajectory


@dataclass
class Batch:
    tokens: torch.Tensor        # (B, T) long, right-padded with <pad>
    action_mask: torch.Tensor   # (B, T) float: 1 on sampled action tokens
    old_logprobs: torch.Tensor  # (B, T) float: log pi_old on action tokens
    step_of_token: torch.Tensor # (B, T) long: step index for action tokens, -1 elsewhere

    def to(self, device: str) -> "Batch":
        return Batch(*(t.to(device) for t in (self.tokens, self.action_mask, self.old_logprobs,
                                               self.step_of_token)))


def collate(trajs: list[Trajectory], pad: int) -> Batch:
    T = max(len(t.tokens) for t in trajs)
    B = len(trajs)
    tokens = torch.full((B, T), pad, dtype=torch.long)
    mask = torch.zeros(B, T)
    old = torch.zeros(B, T)
    step_of = torch.full((B, T), -1, dtype=torch.long)
    for i, t in enumerate(trajs):
        n = len(t.tokens)
        tokens[i, :n] = torch.tensor(t.tokens)
        mask[i, :n] = torch.tensor(t.action_mask, dtype=torch.float)
        old[i, :n] = torch.tensor(t.logprobs)
        step_of[i, :n] = torch.tensor(t.step_of_token)
    return Batch(tokens, mask, old, step_of)
