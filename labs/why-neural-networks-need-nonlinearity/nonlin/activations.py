"""Every activation function in the post, written from its formula.

Each function maps a tensor to a tensor of the same shape, elementwise. The tests check each one against the
PyTorch built-in where one exists. Derivatives come from autograd (see `derivative`), so the dashed f'(x) curves
of Fig 01 and Fig 05 are what the optimizer would really see.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

Act = Callable[[torch.Tensor], torch.Tensor]

SQRT2 = math.sqrt(2.0)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


# The saturating era.
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def tanh(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * sigmoid(2.0 * x) - 1.0  # tanh is a shifted, rescaled sigmoid


# The rectifier family: identity on the positive half-line, arguments about the negative half.
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, x, torch.zeros_like(x))


def leaky_relu(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    return torch.where(x > 0, x, alpha * x)


def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return torch.where(x > 0, x, alpha * torch.expm1(torch.clamp(x, max=0.0)))


SELU_LAMBDA = 1.0507009873554805
SELU_ALPHA = 1.6732632423543772


def selu(x: torch.Tensor) -> torch.Tensor:
    return SELU_LAMBDA * elu(x, SELU_ALPHA)


def relu6(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 6.0)


def relu2(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU (Primer, Nemotron-4). Equal to x * relu(x): value and gate tied to one projection."""
    r = relu(x)
    return r * r


# Smooth self-gates: x times a data-dependent number in [0, 1].
def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU: x * Phi(x). This is PyTorch's nn.GELU() default."""
    return x * normal_cdf(x)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """The tanh approximation used by the original BERT code, GPT-2 and Gemma."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """OpenAI CLIP's activation: x * sigmoid(1.702 x)."""
    return x * sigmoid(1.702 * x)


def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return x * sigmoid(beta * x)


def silu(x: torch.Tensor) -> torch.Tensor:
    return swish(x, 1.0)


def softplus(x: torch.Tensor) -> torch.Tensor:
    # log(1 + e^x), written so it does not overflow for large x.
    return torch.clamp(x, min=0.0) + torch.log1p(torch.exp(-torch.abs(x)))


def mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(softplus(x))


def hard_swish(x: torch.Tensor) -> torch.Tensor:
    """MobileNetV3: the sigmoid replaced by a clamp, so no exponential."""
    return x * relu6(x + 3.0) / 6.0


# A polynomial "activation". The post's section 01 lists it next to sigmoid and ReLU; the lab tests that claim.
def cubic(x: torch.Tensor) -> torch.Tensor:
    return x**3 - x


ACTIVATIONS: dict[str, Act] = {
    "identity": identity,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "leaky_relu_0.1": lambda x: leaky_relu(x, 0.1),  # the slope Fig 03 uses
    "elu": elu,
    "selu": selu,
    "relu6": relu6,
    "relu2": relu2,
    "gelu": gelu,
    "gelu_tanh": gelu_tanh,
    "quick_gelu": quick_gelu,
    "silu": silu,
    "mish": mish,
    "hard_swish": hard_swish,
    "softplus": softplus,
    "cubic": cubic,
}


def get(name: str) -> Act:
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}") from None


class Activation(torch.nn.Module):
    """Wrap a function from this file as a module, so it can sit inside nn.Sequential."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.fn = get(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)

    def extra_repr(self) -> str:
        return self.name


def derivative(fn: Act, x: torch.Tensor) -> torch.Tensor:
    """f'(x) by autograd, elementwise. At ReLU's kink this returns the subgradient PyTorch picks."""
    x = x.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(fn(x).sum(), x)
    return g


def stochastic_gate_sample(x: torch.Tensor, n_draws: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Monte-Carlo mean of x * Bernoulli(Phi(x)) over n_draws coin flips per point (Fig 04).

    Hendrycks and Gimpel's derivation: keep x with probability Phi(x). The expectation is x * Phi(x) = GELU(x).
    """
    p = normal_cdf(x).expand(n_draws, *x.shape).contiguous()
    keep = torch.bernoulli(p, generator=generator)
    return (x * keep).mean(dim=0)
