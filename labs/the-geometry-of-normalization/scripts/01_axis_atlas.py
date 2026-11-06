"""Section 02: every norm is a choice of axes. Build five operators from one function and check them against torch."""

import _setup  # noqa: F401
import torch
import torch.nn.functional as F

from normgeo.ops import AXES, atlas
from normgeo.report import Report

torch.manual_seed(0)
N, C, S, G, EPS = 8, 6, 5, 2, 1e-5
x = torch.randn(N, C, S) * 3 + 1

reference = {
    "batchnorm": lambda t: F.batch_norm(t, None, None, training=True, eps=EPS),
    "layernorm": lambda t: F.layer_norm(t.transpose(1, 2), (C,), eps=EPS).transpose(1, 2),
    "instancenorm": lambda t: F.instance_norm(t, eps=EPS),
    "groupnorm": lambda t: F.group_norm(t, G, eps=EPS),
    "rmsnorm": lambda t: F.rms_norm(t.transpose(1, 2), (C,), eps=EPS).transpose(1, 2),
}
n_stats = {
    "batchnorm": C,
    "layernorm": N * S,
    "instancenorm": N * C,
    "groupnorm": N * G,
    "rmsnorm": N * S,
}

with Report("01_axis_atlas") as say:
    say(f"x has shape (N, C, S) = ({N}, {C}, {S}). One function, normalize(x, dims), builds every operator.")
    say()
    say(f"{'operator':<14}{'axes':<10}{'statistics':>11}{'max |ours - torch|':>20}{'sample 0 moves when sample 1 changes':>40}")
    x2 = x.clone()
    x2[1] = x2[1] * 10 + 5  # change only sample 1
    for kind, ref in reference.items():
        ours = atlas(x, kind, groups=G, eps=EPS)
        err = (ours - ref(x)).abs().max().item()
        moved = (atlas(x2, kind, groups=G, eps=EPS)[0] - ours[0]).abs().max().item()
        axes = "(c-grp, s)" if kind == "groupnorm" else str(AXES[kind]).replace(",)", ")")
        say(f"{kind:<14}{axes:<10}{n_stats[kind]:>11}{err:>20.1e}{moved:>40.3f}")
    say()
    say("Only BatchNorm's output for one sample depends on the other samples in the batch.")
