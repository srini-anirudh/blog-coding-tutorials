"""Gradient flow at initialisation: plain stacks against residual stacks (Figure 12).

A loss gradient is sent back through stacks of 3x3 convs of growing depth. We record the gradient norm that
reaches the first layer's weights, relative to the last layer's. Four variants:
  plain, no BN, PyTorch default init  -- the Jacobians shrink the signal
  plain, no BN, He (Kaiming) init     -- the init that keeps the variance near 1
  plain, with BN                      -- the setting of He et al.'s degradation experiment
  residual, with BN                   -- y = F(x) + x
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from cnnverse.report import Report, set_threads


class Block(nn.Module):
    def __init__(self, c, bn, residual, init):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1, bias=not bn)
        if init == "he":
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        self.bn = nn.BatchNorm2d(c) if bn else nn.Identity()
        self.residual = residual

    def forward(self, x):
        y = self.bn(self.conv(x))
        return torch.relu(y + x) if self.residual else torch.relu(y)


VARIANTS = {
    "plain, no BN, default init": dict(bn=False, residual=False, init="default"),
    "plain, no BN, He init": dict(bn=False, residual=False, init="he"),
    "plain, BN": dict(bn=True, residual=False, init="default"),
    "residual, BN": dict(bn=True, residual=True, init="default"),
}


def grad_ratio(depth, c=16, seed=0, **kw):
    """|grad of first conv weight| / |grad of last conv weight| for one random batch."""
    torch.manual_seed(seed)
    net = nn.Sequential(*[Block(c, **kw) for _ in range(depth)])
    x = torch.randn(16, c, 16, 16)
    net(x).pow(2).mean().backward()
    first, last = net[0].conv.weight.grad.norm(), net[-1].conv.weight.grad.norm()
    return float(first / last)


def main():
    set_threads()
    R = Report("05_residual_gradients.txt")
    depths = (8, 20, 56, 110)
    R("\n|grad first layer| / |grad last layer| at initialisation, 16 channels, 16x16 inputs, mean of 3 seeds")
    R(f"{'variant':>28} " + " ".join(f"{'depth ' + str(d):>10}" for d in depths))
    for name, kw in VARIANTS.items():
        vals = [sum(grad_ratio(d, seed=s, **kw) for s in range(3)) / 3 for d in depths]
        R(f"{name:>28} " + " ".join(f"{v:>10.2e}" for v in vals))
    R("\nA ratio near 1 means the first layer gets as much gradient as the last. Without BN and with the default")
    R("init, the signal vanishes on the way back. He init keeps it within an order of magnitude. With BN, the plain")
    R("stack does not lose the gradient at all: it grows toward the input, by orders of magnitude at depth 56 and 110")
    R("(BN makes deep plain nets explode at initialisation, as Yang et al. 2019 analyse). The residual stack is the")
    R("only one that stays near 1 at every depth. He et al. noted that vanishing gradients do not explain the")
    R("degradation of deep plain nets with BN; 06_degradation measures that degradation directly.")
    R.close()


if __name__ == "__main__":
    main()
