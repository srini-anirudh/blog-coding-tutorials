"""Generation-based evaluation, one metric family per task.

    caption    exact match of the full caption (our captions are deterministic, so this is fair)
    vqa        VQA exact match after normalisation
    ocr        CER and WER (lower is better), plus exact match
    doc        ANLS (DocVQA), plus exact match
    chart      relaxed accuracy (ChartQA, 5% tolerance)
    grounding  box IoU and accuracy at IoU >= 0.5; point-in-box accuracy for pointing questions
"""

from __future__ import annotations

from collections import defaultdict

import torch

from . import grounding as G
from . import metrics as M
from .data import make_example

HEADLINE = {"caption": "em", "vqa": "em", "ocr": "cer", "doc": "anls", "chart": "relaxed_acc", "grounding": "acc@0.5"}


def score(task: str, pred: str, meta: dict) -> dict[str, float]:
    gold = meta["answer"]
    if task in ("caption", "vqa"):
        return {"em": M.exact_match(pred, gold)}
    if task == "ocr":
        return {"cer": M.cer(pred.strip(), gold), "wer": M.wer(pred.strip(), gold), "em": M.exact_match(pred, gold)}
    if task == "doc":
        return {"anls": M.anls(pred, gold), "em": M.exact_match(pred, gold)}
    if task == "chart":
        return {"relaxed_acc": M.relaxed_accuracy(pred, gold)}
    if task == "grounding":
        w, h = meta["size"]
        if meta["kind"] == "point":
            p = G.decode_point(pred, w, h, meta["fmt"])
            return {"point_acc": 0.0 if p is None else M.point_in_box(p, meta["box"]), "parse_fail": float(p is None)}
        box = G.decode_box(pred, w, h, meta["fmt"])
        iou = 0.0 if box is None else M.box_iou(box, meta["box"])
        return {"iou": iou, "acc@0.5": float(iou >= 0.5), "parse_fail": float(box is None)}
    raise ValueError(task)


@torch.no_grad()
def evaluate(model, processor, device: torch.device, tasks: list[str], n: int = 64, split: str = "test",
             batch_size: int = 8, max_new_tokens: int = 24, grounding_format: str = "tokens",
             keep_examples: int = 3) -> dict:
    """Greedy-decode `n` held-out samples per task and average each metric."""
    was_training = model.training
    model.eval()
    results: dict = {}
    examples: dict = {}
    for task in tasks:
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        ex = []
        for start in range(0, n, batch_size):
            samples = [make_example(task, i, split, grounding_format) for i in range(start, min(n, start + batch_size))]
            batch = processor(samples, for_generation=True)
            out = model.generate(batch["input_ids"].to(device),
                                 None if batch["pixel_values"] is None else batch["pixel_values"].to(device),
                                 batch["attention_mask"].to(device), max_new_tokens=max_new_tokens,
                                 do_sample=False, eos_token_id=processor.eos_id, pad_token_id=processor.pad_id)
            preds = processor.tok.batch_decode(out, skip_special_tokens=False)
            for s, pred in zip(samples, preds):
                pred = pred.split("<|im_end|>")[0].replace("<|endoftext|>", "").strip()
                for k, v in score(task, pred, s["meta"]).items():
                    sums[k] += v
                    counts[k] += 1
                if len(ex) < keep_examples:
                    ex.append({"question": s["messages"][0]["content"][-1]["text"], "gold": s["meta"]["answer"],
                               "pred": pred})
        results[task] = {k: sums[k] / counts[k] for k in sums}
        examples[task] = ex
    model.train(was_training)
    return {"metrics": results, "examples": examples}


def format_results(res: dict) -> str:
    lines = [f"{'task':<11}{'metric':<13}{'value':>8}   other"]
    for task, m in res["metrics"].items():
        head = HEADLINE[task]
        other = ", ".join(f"{k}={v:.3f}" for k, v in m.items() if k != head)
        lines.append(f"{task:<11}{head:<13}{m.get(head, float('nan')):>8.3f}   {other}")
    return "\n".join(lines)
