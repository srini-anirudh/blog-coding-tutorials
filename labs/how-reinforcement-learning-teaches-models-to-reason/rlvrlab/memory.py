"""Section 4 of the post: the persistent state PPO-based RLHF holds, against GRPO with a verifier.

A trained model costs about 16 bytes per parameter (bf16 weights 2 + bf16 grads 2 + fp32 master 4 + Adam m 4 +
Adam v 4). A frozen model in bf16 costs 2. `measure_bytes_per_param` checks the trained figure on a real module.
"""

from __future__ import annotations

import torch

TRAINED_BYTES = 16
FROZEN_BYTES = 2

PPO_RLHF = {"policy": True, "critic": True, "reference": False, "reward_model": False}
GRPO_RLVR = {"policy": True, "reference": False}  # the reward is a program; the baseline is a group mean


def persistent_state_gb(models: dict[str, bool], params: float) -> dict[str, float]:
    """GB (1e9 bytes) per model and in total, for models of `params` parameters each. True means trained."""
    out = {name: params * (TRAINED_BYTES if trained else FROZEN_BYTES) / 1e9 for name, trained in models.items()}
    out["total"] = sum(out.values())
    return out


def measure_bytes_per_param(model: torch.nn.Module, mixed_precision: bool) -> float:
    """Take one AdamW step and count the bytes of every tensor the training state keeps.

    fp32 (mixed_precision=False): weights 4 + grads 4 + Adam moments 8 = 16.
    mixed precision: bf16 working weights 2 + bf16 grads 2 + fp32 master weights 4 + Adam moments 8 = 16.
    The bf16 copy is built by hand here, the way mixed-precision trainers keep one.
    """
    master = [p for p in model.parameters()]
    opt = torch.optim.AdamW(master, lr=1e-3)
    n = sum(p.numel() for p in master)
    if mixed_precision:
        work = [p.detach().to(torch.bfloat16).requires_grad_(True) for p in master]
        x = torch.randn(4, master[0].shape[1], dtype=torch.bfloat16)
        torch.func.functional_call(model, dict(zip([k for k, _ in model.named_parameters()], work)), (x,)).sum().backward()
        for p, w in zip(master, work):
            p.grad = w.grad.float()  # the fp32 grad is transient: freed after the step, so not counted
        opt.step()
        kept = [w for w in work] + [w.grad for w in work] + master
    else:
        model(torch.randn(4, master[0].shape[1])).sum().backward()
        opt.step()
        kept = master + [p.grad for p in master]
    for state in opt.state.values():
        kept += [v for k, v in state.items() if torch.is_tensor(v) and k != "step"]
    return sum(t.numel() * t.element_size() for t in kept) / n
