"""Run one training stage from a YAML config, with key=value overrides.

    python scripts/train.py configs/stage1_align.yaml
    python scripts/train.py configs/stage2_mixture.yaml name=s2_noreplay replay_ratio=0
    python scripts/train.py configs/stage2_mixture.yaml lrs.lm=5e-5 freeze=partial vision_layers_to_train=4
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmtrain.train import TrainConfig, run  # noqa: E402


def parse_overrides(args: list[str]) -> dict:
    out: dict = {}
    for a in args:
        key, _, val = a.partition("=")
        node = out
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(val)
    return out


def merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        base[k] = merge(dict(base.get(k) or {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def main() -> None:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml") else {}
    rest = sys.argv[2:] if cfg or (len(sys.argv) > 1 and sys.argv[1].endswith(".yaml")) else sys.argv[1:]
    cfg = merge(cfg, parse_overrides(rest))
    defaults = TrainConfig()
    if "lrs" in cfg:  # keep default LRs for modules the config does not mention
        cfg["lrs"] = {**defaults.lrs, **cfg["lrs"]}
    run(TrainConfig(**cfg))


if __name__ == "__main__":
    main()
