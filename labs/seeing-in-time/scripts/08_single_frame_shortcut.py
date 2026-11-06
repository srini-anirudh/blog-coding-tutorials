"""The single-frame shortcut: which answers survive one frame, shuffling and reversal.

Three questions about a short clip: the ball's colour (one frame suffices), its direction (needs two
ordered frames) and the order of two flashes (needs order). Two models:
  bag   - encodes each frame and averages: it has no notion of order
  video - a tiny transformer with frame-index positions
Part 1 trains both on balanced data and scores them under the four controls (post Figure 10).
Part 2 plants a static shortcut: in the training data red balls always go right and blue balls left.
  The model then answers direction from colour, which a counterfactual test exposes.
Part 3 computes the T-GRPO temporal bonus from sampled answers. No reinforcement learning is run.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from _common import Report
from seeingtime.controls import CONTROLS, accuracy_under_controls, apply_control, t_grpo_bonus, temporal_part, train
from seeingtime.models import BagOfFrames, TinyVideoTransformer
from seeingtime.synth import qa_batch

HEADS = {"colour": 3, "direction": 2, "order": 2}
IN_DIM = 3 * 12 * 12


def make(kind: str):
    return BagOfFrames(IN_DIM, HEADS) if kind == "bag" else TinyVideoTransformer(IN_DIM, HEADS, "index")


def fit(kind: str, bias: float, steps: int, seed: int):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = make(kind)

    def batch_fn(step):
        x, y = qa_batch(128, bias=bias, g=g)
        return (x,), y

    loss = lambda out, y: sum(F.cross_entropy(out[q], y[q]) for q in HEADS)
    train(model, batch_fn, steps, lr=2e-3, loss_fn=loss)
    return model.eval()


def control_rows(scores: dict, label: str) -> list[list]:
    rows = []
    for q in HEADS:
        rows.append([label, q] + [f"{scores[c][q]:.2f}" for c in CONTROLS] + [f"{temporal_part(scores, q):+.2f}"])
    return rows


@torch.no_grad()
def tgrpo_gate_rate(model, frames, labels, group: int = 8, seed: int = 0) -> dict:
    """Share of questions whose sampled answers earn the temporal bonus (ordered beats shuffled)."""
    g = torch.Generator().manual_seed(seed)
    ordered = model(frames)
    shuffled = model(apply_control(frames, "shuffled", g))
    out = {}
    for q, y in labels.items():
        p_o, p_s = ordered[q].softmax(-1), shuffled[q].softmax(-1)
        gated = 0
        for i in range(len(y)):
            a_o = torch.multinomial(p_o[i], group, replacement=True, generator=g) == y[i]
            a_s = torch.multinomial(p_s[i], group, replacement=True, generator=g) == y[i]
            gated += int(t_grpo_bonus(a_o, a_s).sum() > 0)
        out[q] = gated / len(y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rep = Report("08_single_frame_shortcut")
    test_x, test_y = qa_batch(3000, bias=0.0, g=torch.Generator().manual_seed(123))
    header = ["model", "question"] + list(CONTROLS) + ["full - best(one, shuffled)"]
    rep.print(f"Accuracy on 3,000 balanced test clips. Chance: colour 0.33, direction 0.50, order 0.50. "
              f"{args.steps} steps, batch 128, seed {args.seed}.")

    rep.print("\n== Part 1: balanced training data ==")
    models, rows = {}, []
    for kind in ("bag", "video"):
        models[kind] = fit(kind, 0.0, args.steps, args.seed)
        s = accuracy_under_controls(models[kind], test_x, test_y)
        rep.data[f"balanced_{kind}"] = s
        rows += control_rows(s, kind)
    rep.table(header, rows)
    rep.print("Reversal flips direction and order, so a model that uses time scores near 0 there, not 0.5.")

    rep.print("\n== Part 2: training data with a static shortcut (red always right, blue always left) ==")
    biased = fit("video", 1.0, args.steps, args.seed)
    bx, by = qa_batch(3000, bias=1.0, g=torch.Generator().manual_seed(124))
    rows = []
    for name, (x, y) in (("biased test (same shortcut)", (bx, by)), ("balanced test (counterfactual)", (test_x, test_y))):
        s = accuracy_under_controls(biased, x, y)
        rep.data[f"shortcut_{name}"] = s
        rows += control_rows(s, name)
    rep.table(["test set"] + header[1:], rows)
    colour = test_y["colour"]
    with torch.no_grad():
        pred = biased(test_x)["direction"].argmax(-1)
    rows = []
    for c, name in ((0, "red"), (1, "green"), (2, "blue")):
        m = colour == c
        rows.append([name, f"{float((pred[m] == test_y['direction'][m]).float().mean()):.2f}",
                     f"{float(pred[m].float().mean()):.2f}"])
    rep.print("\nShortcut model, direction on the balanced test, split by colour:")
    rep.table(["ball colour", "direction accuracy", "share answered 'right'"], rows)
    rep.data["shortcut_by_colour"] = rows

    rep.print("\n== Part 3: T-GRPO temporal bonus (8 sampled answers per question, ordered vs shuffled) ==")
    rows = []
    for name, model in (("bag", models["bag"]), ("video", models["video"]), ("video, shortcut-trained", biased)):
        rates = tgrpo_gate_rate(model, test_x[:500], {q: y[:500] for q, y in test_y.items()})
        rows.append([name] + [f"{rates[q]:.2f}" for q in HEADS])
        rep.data.setdefault("tgrpo_gate_rate", {})[name] = rates
    rep.table(["model", "colour", "direction", "order"], rows)
    rep.print("Share of questions where ordered answers beat shuffled ones, so correct answers earn the bonus.\n"
              "Colour is answered as well from shuffled frames, so ordered answers almost never beat them.\n"
              "The bag model has no order, so its direction and order rates are what sampling noise alone gives.")
    rep.save()


if __name__ == "__main__":
    main()
