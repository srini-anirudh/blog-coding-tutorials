"""Receptive fields: the recursion, the measured footprint, and the smaller effective field.

For each stacking strategy of Figure 04, the theoretical receptive field from the recursion is compared with
the width of the input region whose gradient is nonzero. Then, with random weights, how much of the gradient
falls in the centre of that region (Luo et al., 2017).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cnnverse import conv, receptive as rf
from cnnverse.report import Report, set_threads
from cnnverse.zoo import NINE
from cnnverse.profile import profile


def stem_layers(model_name):
    """(k, s, d) of every conv and pool along the main path, read from a profiled model."""
    import torch.nn as nn
    b, inp, *_ = NINE[model_name]
    m = b()
    out = []
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.MaxPool2d)):
            k = mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0]
            s = mod.stride if isinstance(mod.stride, int) else mod.stride[0]
            out.append((k, s, 1))
    return out


def main():
    set_threads()
    R = Report("02_receptive_fields.txt")

    R("\n== Recursion vs measured footprint (weights all 1, gradient of the centre output unit) ==")
    R(f"{'strategy':>36} {'params at C=64':>15} {'jump':>5} {'RF recursion':>13} {'RF measured':>12}")
    for name, layers in rf.STRATEGIES.items():
        j, r = rf.rf_recursion(layers)[-1]
        size = r + 2 * j * 4 + 1  # leave room so the footprint does not touch the border
        size += (size % 2 == 0)
        gmap = rf.gradient_map(rf.conv_stack(layers), size)
        params = sum(conv.conv_params(k, 64, 64) for k, _, _ in layers)
        R(f"{name:>36} {params:>15,} {j:>5} {r:>13} {rf.support_width(gmap):>12}")

    R("\nWith stride 1, n layers of 3x3 give 2n+1:")
    R("  " + ", ".join(f"n={n}: {rf.rf_recursion([(3, 1, 1)] * n)[-1][1]}" for n in (1, 2, 4, 8, 16, 32)))
    R("With a stride-2 layer every second layer, growth compounds:")
    R("  " + ", ".join(f"n={n}: {rf.rf_recursion([(3, 2 if i % 2 else 1, 1) for i in range(n)])[-1][1]}"
                       for n in (2, 4, 6, 8, 10)))

    R("\n== Effective receptive field: random weights, ReLU, 8 layers of 3x3 (theoretical RF 17) ==")
    layers = [(3, 1, 1)] * 8
    acc = None
    for seed in range(20):
        gmap = rf.gradient_map(rf.conv_stack(layers, channels=8, act=True), 33, channels=8, weights="random", seed=seed)
        acc = gmap if acc is None else acc + gmap
    row = acc[16] / acc[16].max()
    R("normalised |gradient| along the centre row, offsets -8..8:")
    R("  " + " ".join(f"{v:.2f}" for v in row[8:25].tolist()))
    for hw in (1, 2, 4, 8):
        R(f"  share of gradient within the central {2 * hw + 1}x{2 * hw + 1} square"
          f" ({(2 * hw + 1) ** 2 / 17 ** 2:.0%} of the RF area): {rf.mass_within(acc, hw):.2f}")

    R("\n== The theoretical receptive field of real stems (main path, from the zoo specs) ==")
    for name in ("AlexNet", "VGG-16"):
        lay = stem_layers(name)
        j, r = rf.rf_recursion(lay)[-1]
        R(f"{name}: {len(lay)} conv/pool layers, final jump {j} px, theoretical RF {r} px")
    R.close()


if __name__ == "__main__":
    main()
