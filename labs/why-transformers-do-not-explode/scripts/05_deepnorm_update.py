"""Part V: the size of one update, ||Delta F||, against depth for Post-LN, DeepNorm and Pre-LN.

    python labs/why-transformers-do-not-explode/scripts/05_deepnorm_update.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from stablelab.common import Report, seed_all  # noqa: E402
from stablelab.model import CharData, Config, TinyTransformer, deepnorm_constants, load_corpus  # noqa: E402


def update_size(model: TinyTransformer, data: CharData, lr: float, steps: int = 1, seed: int = 0) -> float:
    """Mean per-token ||F(x; theta_after) - F(x; theta_before)|| of the final hidden state after `steps` Adam steps."""
    gen = torch.Generator().manual_seed(seed)
    probe, _ = data.batch(16, torch.Generator().manual_seed(999))
    with torch.no_grad():
        before = model.hidden(probe)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = data.batch(16, gen)
        opt.zero_grad()
        model.loss(x, y).backward()
        opt.step()
    with torch.no_grad():
        after = model.hidden(probe)
    return (after - before).norm(dim=-1).mean().item()


def main() -> None:
    seed_all(0)
    r = Report("05_deepnorm_update")
    r("1. DeepNorm constants for a decoder-only stack of M layers: alpha = (2M)^(1/4), beta = (8M)^(-1/4).")
    r.table(["M", "alpha", "beta"], [[m, *deepnorm_constants(m)] for m in (6, 12, 48, 100, 500, 1000)])
    r()

    data = CharData(load_corpus(), seq=64)
    lr = 1e-4
    r(f"2. ||Delta F|| after one Adam step (lr {lr}, batch 16 x 64): mean per-token change of the final hidden state")
    r("   (after the last LN, so every model's output has norm about sqrt(d) = 8). d = 64, 4 heads.")
    r("   Post-LN uses Xavier init (DeepNet's baseline); DeepNorm uses Xavier with beta and alpha on the rail.")
    variants = [("Post-LN", "post", "xavier"), ("DeepNorm", "deep", "xavier"),
                ("Pre-LN", "pre", "default"), ("Pre-LN, 1/sqrt(2L)", "pre", "scaled")]
    depths = (3, 6, 12, 24, 48)
    rows = []
    for name, norm, init in variants:
        sizes = []
        for L in depths:
            torch.manual_seed(0)
            model = TinyTransformer(Config(vocab=data.vocab, d=64, layers=L, norm=norm, init=init))
            sizes.append(update_size(model, data, lr))
        rows.append([name] + sizes + [sizes[-1] / sizes[0]])
        print(f"  {name}: {sizes}", flush=True)
    r.table(["model"] + [f"L={L}" for L in depths] + ["L=48 / L=3"], rows)
    r("With Adam every parameter moves by about lr per coordinate, so ||Delta theta|| grows with the parameter count")
    r("and the question is how much of that reaches the output. DeepNet's bound says vanilla Post-LN passes it")
    r("through roughly in proportion to the number of sublayers, while DeepNorm's alpha and beta keep it flat.")
    post, deep = rows[0], rows[1]
    r(f"Measured: depth grows 16x. Post-LN's update grows {post[-1]:.1f}x, about linear in depth. At L = 48 one step")
    r(f"moves the output by {post[-2]:.1f}, more than the output's own norm of 8: a single step rewrites the function.")
    r(f"DeepNorm's update is {post[1] / deep[1]:.0f}x smaller at L = 3 and grows {deep[-1]:.1f}x: much flatter, not exactly")
    r("flat at this scale. Pre-LN's final LN bounds its output, so its growth slows at depth.")
    r("This measures the bound's quantity on a tiny model; it is not DeepNet's 1,000-layer experiment.")
    r.save()


if __name__ == "__main__":
    main()
