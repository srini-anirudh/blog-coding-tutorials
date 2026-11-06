"""scratchnet: a small neural-network training project in raw PyTorch.

Modules, in the order the tutorial builds them:
    tensors        storage, strides, offsets and views
    broadcasting   the broadcasting rule and stride-0 expansion
    device         device choice and synchronisation
    matmul         matrix multiplication from loops to kernels
    autograd_tools walking the grad_fn graph
    layers         Linear, ReLU, Dropout, Standardize, MLP
    losses         MSE, log_softmax and cross entropy
    optim          SGD, AdamW, gradient norm and clipping
    data           Dataset classes and collate functions
    train          the training loop, evaluation and checkpoints
    debug          hooks for activation stats, grad norms and NaNs
"""

from .device import pick_device, synchronize
from .layers import MLP, Dropout, Linear, ReLU, Standardize, count_parameters
from .losses import cross_entropy, log_softmax, mse_loss
from .optim import SGD, AdamW, clip_grad_norm, grad_norm
from .train import TrainConfig, fit

__all__ = [
    "MLP", "SGD", "AdamW", "Dropout", "Linear", "ReLU", "Standardize", "TrainConfig",
    "clip_grad_norm", "count_parameters", "cross_entropy", "fit", "grad_norm",
    "log_softmax", "mse_loss", "pick_device", "synchronize",
]
