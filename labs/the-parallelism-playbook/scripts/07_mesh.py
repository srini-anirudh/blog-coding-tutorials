"""The device mesh: rank -> (tp, cp, dp, pp) with TP fastest-varying, and which groups leave an 8-GPU node."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppbook.dist_utils import Report  # noqa: E402
from ppbook.mesh import coords, placement_report  # noqa: E402


def main():
    rep = Report("07_mesh.txt")
    rep("Pure arithmetic, no processes. 8 GPUs per node; ranks r and r+1 are on the same node unless r+1 is a")
    rep("multiple of 8.")
    rep("")
    for r in (0, 7, 8, 63, 64, 511):
        rep(f"  rank {r:>3} in 512 = TP 8 x CP 1 x DP 8 x PP 8 -> {coords(r, 8, 1, 8, 8)}")
    rep("")
    configs = [("TP 8, DP 8, PP 8 (512)", (8, 1, 8, 8)), ("TP 16, DP 4, PP 8 (512)", (16, 1, 4, 8)),
               ("TP 4, CP 2, DP 8, PP 8 (512)", (4, 2, 8, 8)), ("TP 2, DP 32 (64)", (2, 1, 32, 1))]
    for name, (tp, cp, dp, pp) in configs:
        rep(name)
        for axis, info in placement_report(tp, cp, dp, pp).items():
            if info["size"] == 1:
                continue
            ex = info["example"]
            ex = ex if len(ex) <= 8 else ex[:4] + ["..."] + ex[-2:]
            rep(f"  {axis}: {info['groups']:>3} groups of {info['size']:>2}, {info['cross_node']:>3} leave a node;"
                f" e.g. {ex}")
        rep("")
    rep("With TP fastest-varying, every TP group of size <= 8 stays in one node, so its 4 AllReduces per layer")
    rep("ride the fast fabric. TP 16 forces every TP group across two nodes. DP and PP groups cross nodes, which")
    rep("is where their infrequent, overlappable traffic belongs.")
    rep.save()


if __name__ == "__main__":
    main()
