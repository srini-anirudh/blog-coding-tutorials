"""A layer-by-layer spec engine: parameters, MACs and activations, read off real nn.Modules.

MACs count only convolutions and linear layers, the convention the papers use. Profiling runs on the
`meta` device, so no arithmetic is done and VGG-16 at 224 px costs nothing to profile.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn


def n_params(module, recurse=True):
    """Trainable parameters. A masked conv (LeNet's sparse C3) counts only its unmasked weights."""
    total = 0
    mods = module.modules() if recurse else [module]
    for m in mods:
        for name, p in m.named_parameters(recurse=False):
            if name == "weight" and hasattr(m, "n_weights"):
                total += m.n_weights
            else:
                total += p.numel()
    return total


def layer_macs(m, out):
    if isinstance(m, nn.Conv2d):
        positions = out.numel() // out.shape[0] // m.out_channels
        if hasattr(m, "n_weights"):
            return positions * m.n_weights
        return positions * m.out_channels * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
    if isinstance(m, nn.Linear):
        return out.numel() // out.shape[0] * m.in_features
    return 0


@dataclass
class Profile:
    name: str
    params: int = 0
    macs: int = 0
    act_elems: int = 0
    layers: list = field(default_factory=list)   # (module path, type, out shape, params, macs)

    def by_prefix(self, depth=1):
        """Sum MACs and params by the first `depth` parts of the module path (stage-level profile)."""
        agg = {}
        for path, _, _, p, m in self.layers:
            key = ".".join(path.split(".")[:depth])
            a = agg.setdefault(key, [0, 0])
            a[0] += p
            a[1] += m
        return agg


COUNTED = (nn.Conv2d, nn.Linear)
ACT_LAYERS = (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)


def profile(model, input_size, name="", device="meta"):
    """Run one image through `model` on `device` and record every conv, linear and pooling layer."""
    model = model.to(device).eval()
    prof = Profile(name, params=n_params(model))
    hooks = []

    def hook(path):
        def fn(m, inp, out):
            mac = layer_macs(m, out)
            prof.macs += mac
            prof.act_elems += out.numel() // out.shape[0]
            prof.layers.append((path, type(m).__name__, tuple(out.shape[1:]), n_params(m, recurse=False), mac))
        return fn

    for path, m in model.named_modules():
        if isinstance(m, ACT_LAYERS):
            hooks.append(m.register_forward_hook(hook(path)))
    with torch.no_grad():
        model(torch.zeros(1, *input_size, device=device))
    for h in hooks:
        h.remove()
    return prof


def head_param_share(model):
    """Share of parameters in the classifier: the Linear layers that come after the last Flatten."""
    mods = list(model.modules())
    last_flat = max(i for i, m in enumerate(mods) if isinstance(m, nn.Flatten))
    head = sum(n_params(m, recurse=False) for m in mods[last_flat:] if isinstance(m, nn.Linear))
    return head / n_params(model)


def saved_activation_bytes(model, x, loss_fn=None):
    """Bytes autograd keeps alive for the backward pass after one forward, measured with saved-tensor hooks.

    Parameters are excluded (they are counted as weights). Tensors that share storage are counted once.
    """
    param_ptrs = {p.data_ptr() for p in model.parameters()}
    seen = {}

    def pack(t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in param_ptrs and ptr not in seen:
            seen[ptr] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(x)
        loss = loss_fn(out) if loss_fn else out.float().pow(2).mean()
    total = sum(seen.values())
    del loss, out
    return total
