"""Hooks for looking inside a training step without editing the model.

A forward hook runs after a module's forward and sees its output.
A post-accumulate-grad hook runs during backward, right after a parameter's
.grad has been written. Both return a handle; call .remove() when done.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


class HookSet:
    """Owns a list of hook handles and removes them all on exit."""

    def __init__(self) -> None:
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


@dataclass
class ActStat:
    mean: float
    std: float
    frac_zero: float  # after a ReLU, the share of units that output exactly 0


class ActivationStats(HookSet):
    """Record mean, std and fraction of zeros of every matching module's output."""

    def __init__(self, model: nn.Module, types: tuple[type, ...]) -> None:
        super().__init__()
        self.stats: dict[str, ActStat] = {}
        for name, module in model.named_modules():
            if isinstance(module, types):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            out = output.detach().float()
            self.stats[name] = ActStat(out.mean().item(), out.std().item(), (out == 0).float().mean().item())

        return hook


@dataclass
class GradRecord:
    order: list[str] = field(default_factory=list)  # names, in the order their grads became ready
    norms: dict[str, torch.Tensor] = field(default_factory=dict)


class GradNormMonitor(HookSet):
    """Record each parameter's gradient norm the moment backward finishes writing it."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.record = GradRecord()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.handles.append(p.register_post_accumulate_grad_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(p: torch.Tensor) -> None:
            self.record.order.append(name)
            # Keep a tensor, not .item(): .item() would force a GPU sync inside backward.
            self.record.norms[name] = p.grad.detach().norm()

        return hook

    def reset(self) -> None:
        self.record = GradRecord()


class NaNDetector(HookSet):
    """Raise at the first module whose forward output, or parameter whose grad, is not finite."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        for name, module in model.named_modules():
            self.handles.append(module.register_forward_hook(self._forward_hook(name or "<root>")))
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.handles.append(p.register_post_accumulate_grad_hook(self._grad_hook(name)))

    @staticmethod
    def _forward_hook(name: str):
        def hook(module: nn.Module, inputs: tuple, output) -> None:
            if torch.is_tensor(output) and not torch.isfinite(output).all():
                finite_in = all(torch.isfinite(x).all() for x in inputs if torch.is_tensor(x))
                where = "first appears in" if finite_in else "passes through"
                raise FloatingPointError(f"non-finite output {where} {name} ({type(module).__name__})")

        return hook

    @staticmethod
    def _grad_hook(name: str):
        def hook(p: torch.Tensor) -> None:
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"non-finite gradient for parameter {name}")

        return hook


def param_norms(model: nn.Module) -> dict[str, float]:
    return {name: p.detach().norm().item() for name, p in model.named_parameters()}


def check_same_device(model: nn.Module, **tensors: torch.Tensor) -> None:
    """Fail early, with names, instead of deep inside a matmul."""
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1:
        raise RuntimeError(f"model parameters are split across devices: {devices}")
    (model_device,) = devices
    for name, t in tensors.items():
        if t.device != model_device:
            raise RuntimeError(f"{name} is on {t.device} but the model is on {model_device}; call .to(device)")
