"""Questions 3 and 4 at small scale: where vision joins the LM, and what is frozen.

One pretrained tiny LM (text tasks: cell lookup, majority, addition mod 8). One MLP connector,
16 visual tokens. Six recipes, each trained on the same visual questions with no text replay:

    prefix, LM frozen        LLaVA stage 1: only the connector trains
    prefix, LM trains        the modern default once the LM is unfrozen
    gated xattn, LM frozen   Flamingo: connector + new tanh-gated cross-attention layers
    visual expert, LM frozen CogVLM: connector + a full copy of every projection, used at visual positions
    PLoRA r=8, LM frozen     InternLM-XComposer2: connector + low-rank deltas used at visual positions
    encoder-free, LM trains  Fuyu: raw 4x4 patches, one linear layer, no encoder

We report visual accuracy and the LM's text accuracy before and after. "add" is the text task
that the visual training never touches, so it shows forgetting most clearly.

    python labs/the-vlm-architecture-gallery/scripts/06_fusion_freeze.py --seed 0
    python labs/the-vlm-architecture-gallery/scripts/06_fusion_freeze.py --report
"""

import argparse
import json
import statistics

import _common
import torch

from gallery.fusion import attention_cost
from gallery.model import TinyVLM
from gallery.train import evaluate, fresh_copy, pretrain_lm, text_scores, train_vlm

RECIPES = [
    ("prefix, LM frozen", dict(fusion="prefix"), "connector"),
    ("prefix, LM trains", dict(fusion="prefix"), "all"),
    ("gated xattn, LM frozen", dict(fusion="xattn"), "connector"),
    ("visual expert, LM frozen", dict(fusion="expert"), "connector"),
    ("PLoRA r=8, LM frozen", dict(fusion="expert", expert_rank=8), "connector"),
    ("encoder-free, LM trains", dict(fusion="encfree"), "all"),
]


def run(seed: int, steps: int):
    _common.threads()
    lm = pretrain_lm(seed=seed, steps=1500, cache=_common.RUNS / f"lm_seed{seed}.pt")
    before = text_scores(lm)
    rows = []
    for label, kw, schedule in RECIPES:
        torch.manual_seed(seed)
        vlm = TinyVLM(fresh_copy(lm), connector="mlp", **kw)
        # at init, every recipe must leave text-only behaviour exactly as the pretrained LM had it
        ids = torch.randint(0, 19, (4, 19))
        same_at_init = torch.equal(vlm.lm(ids), lm(ids))
        info = train_vlm(vlm, ("fine", "coarse"), schedule, steps=steps, seed=seed)
        after = text_scores(vlm)
        row = {"recipe": label, "schedule": schedule, "fine": evaluate(vlm, "fine", True),
               "coarse": evaluate(vlm, "coarse", True), "text_before": before, "text_after": after,
               "text_identical_at_init": same_at_init, "lm_seq_len": vlm.lm_sequence_length(3), **info}
        print(json.dumps(row), flush=True)
        rows.append(row)
    out = _common.RESULTS / f"06_fusion_freeze_seed{seed}.json"
    out.write_text(json.dumps({"hardware": _common.hardware(), "command": _common.command(),
                               "steps": steps, "seed": seed, "rows": rows}, indent=1) + "\n")


def report():
    files = sorted(_common.RESULTS.glob("06_fusion_freeze_seed*.json"))
    runs = [json.loads(f.read_text()) for f in files]
    rep = _common.Report("06_fusion_freeze")
    rep(f"# merged from {len(runs)} seeds: " + ", ".join(f.name for f in files))
    rep(f"# each seed run on: {runs[0]['hardware']}; {runs[0]['steps']} steps, batch 128, no text replay")
    rep("mean over seeds (min to max); chance 0.125.")
    for r in runs:
        b = r["rows"][0]["text_before"]
        rep(f"seed {r['seed']}: pretrained LM text accuracy before visual training: "
            + ", ".join(f"{k} {v:.3f}" for k, v in b.items()))
    rep(f"{'recipe':<26}{'trainable':>10}{'seq':>5}{'fine':>8}{'coarse':>8}"
        f"{'text lookup after':>19}{'text add after':>16}")
    merged = []
    for i, (label, _, _) in enumerate(RECIPES):
        rs = [r["rows"][i] for r in runs]
        m = lambda key, sub=None: [r[key][sub] if sub else r[key] for r in rs]  # noqa: E731
        f, c, tl, ta = m("fine"), m("coarse"), m("text_after", "fine"), m("text_after", "add")
        rng = lambda v: f"{statistics.mean(v):.3f}"  # noqa: E731
        rep(f"{label:<26}{rs[0]['trainable_params']:>10,}{rs[0]['lm_seq_len']:>5}{rng(f):>8}{rng(c):>8}"
            f"{rng(tl):>19}{rng(ta):>16}   [fine {min(f):.3f}-{max(f):.3f}, add {min(ta):.3f}-{max(ta):.3f}]")
        merged.append({"recipe": label, "trainable_params": rs[0]["trainable_params"], "fine": f, "coarse": c,
                       "text_lookup_after": tl, "text_add_after": ta,
                       "text_identical_at_init": all(m("text_identical_at_init"))})
    rep("seq = LM sequence length for a 3-token question (visual tokens + text; xattn keeps vision out of the stream).")
    rep("all recipes give bit-identical text logits at init: " + str(all(r["text_identical_at_init"] for r in merged)))
    rep("")
    rep("attention-score cost per sequence for a real-size picture (576 visual + 64 text tokens, d 4096, 32 layers):")
    for fusion in ("prefix", "xattn"):
        c = attention_cost(64, 576, 4096, 32, fusion)
        rep(f"  {fusion:<7} decoder sequence {c['seq_len']:>4}   attention MACs {c['attn_macs'] / 1e9:6.1f} G"
            f"   cached K/V positions {c['kv_positions']:,}")
    rep.save({"rows": merged})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.seed, a.steps)
