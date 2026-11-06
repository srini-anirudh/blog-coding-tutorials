"""CHAIR rewards silence: truncate the same descriptions and watch CHAIR improve while coverage falls.

    python labs/aligning-vision-language-models/scripts/chair_silence.py

The SFT policy describes 600 held-out scenes (greedy). We keep only the first k claims of each description and
score CHAIR_i (hallucinated object mentions / object mentions), CHAIR_s (descriptions with any hallucinated object)
and object recall (present objects mentioned / present objects). k = 0 is an empty caption.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundpref.metrics import score_response, summarise  # noqa: E402
from groundpref.model import sample  # noqa: E402
from groundpref.report import Log  # noqa: E402
from groundpref.train import eval_scenes, load_sft, set_threads  # noqa: E402
from groundpref.world import decode  # noqa: E402


def main() -> None:
    set_threads()
    log = Log("chair_silence")
    model = load_sft(log)
    scenes, imgs = eval_scenes(600, seed=3000)
    claims = [decode(ids)[0] for ids in sample(model, imgs, greedy=True)]
    log(f"{'first k claims':>14s} {'CHAIR_i':>8s} {'CHAIR_s':>8s} {'recall':>8s} {'claims/resp':>12s}")
    out = {}
    for k in (0, 1, 2, 3, 99):
        r = summarise([score_response(s, c[:k]) for s, c in zip(scenes, claims)])
        out["all" if k == 99 else str(k)] = r
        log(f"{'all' if k == 99 else k:>14} {r['chair_i']:>8.3f} {r['chair_s']:>8.3f} {r['recall']:>8.3f} "
            f"{r['claims_per_response']:>12.2f}")
    log.save({"results": out})


if __name__ == "__main__":
    main()
