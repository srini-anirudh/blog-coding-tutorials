"""Cross-entropy cannot tell a near miss from a far miss; an IoU reward can (post Figures 3 and 5)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointlab.coords import digit_error_shift  # noqa: E402
from pointlab.geometry import box_iou  # noqa: E402
from pointlab.report import Report  # noqa: E402
from pointlab.scoring import ce_logit_grad_norm, cross_entropy, edge_iou, expected_iou_reward  # noqa: E402

R = Report("04_ce_blindness")
TRUE = (268, 584)          # the true box spans bins 268 to 584 on a 0-1000 grid; only the right edge is predicted

preds = {"A: confident, correct": {584: 1.0},
         "B: near miss": {583: 0.9, 584: 0.1},
         "C: far miss": {884: 0.9, 584: 0.1}}
rows, data = [], {}
for name, p in preds.items():
    arg = max(p, key=p.get)
    ce = abs(cross_entropy(p, 584))
    rows.append([name, str(p), f"{ce + 0.0:.2f}", f"{ce_logit_grad_norm(p, 584):.3f}", f"{edge_iou(arg, TRUE):.3f}",
                 f"{expected_iou_reward(p, TRUE):.3f}"])
    data[name] = {"ce": ce, "iou_argmax": edge_iou(arg, TRUE), "expected_iou": expected_iou_reward(p, TRUE)}
R("Figure 5: three predictions for the right edge x2 (true bin 584)")
R.table(["prediction", "p(bin)", "cross-entropy", "|dCE/dlogits|", "IoU of argmax box", "expected IoU reward"], rows)
R("B and C have the same loss and the same gradient size. Only the IoU reward ranks them.")

# ---- training one coordinate with each signal ---------------------------------------------------
R("")
R("Start from a uniform distribution over 1,000 bins and take 200 Adam steps (lr 0.05) on the logits")
bins = torch.arange(1000).float()
iou_of_bin = torch.tensor([edge_iou(int(k), TRUE) for k in bins])
rows = []
for objective in ("cross-entropy", "expected IoU reward"):
    logits = torch.zeros(1000, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.05)
    hist = []
    for step in range(201):
        p = logits.softmax(0)
        loss = -p[584].log() if objective == "cross-entropy" else -(p * iou_of_bin).sum()
        if step in (0, 25, 50, 100, 200):
            mass_near = p[(bins - 584).abs() <= 20].sum().item()
            hist.append(f"step {step}: p(584) {p[584].item():.2f}, mass within 20 bins {mass_near:.2f}, "
                        f"E[IoU] {(p * iou_of_bin).sum().item():.3f}")
        opt.zero_grad()
        loss.backward()
        opt.step()
    R(f"  {objective}:")
    for h in hist:
        R("    " + h)
    data[f"train_{objective}"] = hist
R("Cross-entropy pulls mass onto the one true bin; its gradient treats every other bin alike. The IoU reward")
R("pushes mass toward the true bin from both sides, and a near miss already earns almost the full reward.")

# ---- Figure 3: one wrong digit ----------------------------------------------------------------
R("")
R("Figure 3: the box (268, 372, 584, 811) on a 0-1000 grid with one wrong digit in x2")
true_box = (268, 372, 584, 811)
rows = []
for wrong, which in [(884, "hundreds"), (589, "units"), (594, "tens")]:
    b = (268, 372, wrong, 811)
    rows.append([which, f"584 -> {wrong}", f"{digit_error_shift(584, wrong):.1%}", f"{box_iou(b, true_box):.3f}",
                 "1 token"])
R.table(["wrong digit", "x2", "edge moves (share of image)", "box IoU", "cost to cross-entropy"], rows)
R("Each wrong digit is one wrong token to the loss, whatever its place value.")
R.save(data)
