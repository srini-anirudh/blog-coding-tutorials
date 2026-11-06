"""Evaluation controls against the single-frame shortcut, and the T-GRPO temporal reward.

Controls change the frames, never the labels:
  full      - the frames as recorded
  one_frame - only the centre frame, repeated to the same length (so the model's input shape is unchanged)
  shuffled  - frames in a random order
  reversed  - frames played backwards
"""

from __future__ import annotations

import torch

CONTROLS = ("full", "one_frame", "shuffled", "reversed")


def apply_control(frames: torch.Tensor, control: str, g: torch.Generator | None = None) -> torch.Tensor:
    B, T = frames.shape[:2]
    if control == "full":
        return frames
    if control == "one_frame":
        return frames[:, T // 2: T // 2 + 1].expand_as(frames).contiguous()
    if control == "reversed":
        return frames.flip(1)
    if control == "shuffled":
        perm = torch.stack([torch.randperm(T, generator=g) for _ in range(B)])
        return frames[torch.arange(B)[:, None], perm]
    raise ValueError(control)


@torch.no_grad()
def accuracy_under_controls(model, frames, labels: dict[str, torch.Tensor], seed: int = 0) -> dict:
    """{control: {question: accuracy}}."""
    g = torch.Generator().manual_seed(seed)
    model.eval()
    out = {}
    for c in CONTROLS:
        logits = model(apply_control(frames, c, g))
        out[c] = {q: float((logits[q].argmax(-1) == y).float().mean()) for q, y in labels.items()}
    return out


def temporal_part(scores: dict, question: str) -> float:
    """Full score minus the best control that destroys order or motion (one frame or shuffled)."""
    return scores["full"][question] - max(scores["one_frame"][question], scores["shuffled"][question])


def t_grpo_bonus(correct_ordered: torch.Tensor, correct_shuffled: torch.Tensor, alpha: float = 0.3,
                 mu: float = 1.0) -> torch.Tensor:
    """Temporal reward in the style of Video-R1's T-GRPO, for one question.

    correct_ordered / correct_shuffled: 0/1 per sampled answer, from ordered and from shuffled frames.
    If the group's accuracy on ordered frames beats mu times its accuracy on shuffled frames, every correct
    ordered answer gets +alpha. Otherwise nobody gets the bonus. alpha and mu here are this lab's settings;
    check the Video-R1 paper for the values it used.
    """
    p, p_shuf = correct_ordered.float().mean(), correct_shuffled.float().mean()
    gate = (p > mu * p_shuf).float()
    return alpha * gate * correct_ordered.float()


def train(model, batch_fn, steps: int, lr: float = 3e-3, loss_fn=None, log_every: int = 0):
    """A plain AdamW loop. batch_fn(step) -> (inputs tuple, targets); loss_fn(outputs, targets) -> loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    history = []
    model.train()
    for step in range(steps):
        inputs, targets = batch_fn(step)
        loss = loss_fn(model(*inputs), targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            history.append((step, float(loss)))
    return history
