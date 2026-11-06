"""Language forgetting and the levers against it (post Figure 5 and the mitigations table).

The base model is the decoder trained on text-only chat (reverse a word list). Multimodal SFT then
trains on VQA, chat and captions. Delta S_text = text score after SFT minus the base model's score.

Levers: text-only replay at 0%, 10% and 50% of examples; the decoder at 0.1x learning rate; LoRA
(rank 8) on the decoder; a frozen decoder (only encoder and connector train).

    python labs/one-model-many-visual-skills/scripts/04_forgetting.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillmix.evaluate import eval_text  # noqa: E402
from skillmix.runs import base_model, sweep, write_result  # noqa: E402
from skillmix.train import RunConfig  # noqa: E402

MM = {"vqa": 1 / 3, "chat": 1 / 3, "caption": 1 / 3}
SUITE = [("eval_text", {}), ("eval_vqa", {"fmt": True}), ("eval_chat", {}), ("eval_caption", {})]


def mix(replay: float) -> dict:
    return {**{k: v * (1 - replay) for k, v in MM.items()}, **({"text": replay} if replay else {})}


def main() -> None:
    base_text = eval_text(base_model())["text_exact"]
    runs = [("full, replay 0%", "full", 0.0), ("full, replay 10%", "full", 0.1), ("full, replay 50%", "full", 0.5),
            ("LLM lr x0.1, replay 0%", "low_llm_lr", 0.0), ("LoRA r8, replay 0%", "lora", 0.0),
            ("frozen LLM, replay 0%", "frozen_llm", 0.0)]
    res = sweep([(n, RunConfig(mixture=mix(r), strategy=s, fmt=True), SUITE) for n, s, r in runs])
    lines = [f"base model text score (reverse, exact): {base_text:.3f}",
             "multimodal score = mean of VQA exact, chat correct and caption recall", "",
             f"{'run':<24} {'text':>6} {'dS_text':>8} {'VQA':>6} {'chat':>6} {'recall':>7} {'MM score':>9}"]
    for name, r in res.items():
        r["delta_text"] = round(r["text_exact"] - base_text, 3)
        r["mm_score"] = round((r["vqa_exact"] + r["chat_correct"] + r["recall"]) / 3, 3)
        lines.append(f"{name:<24} {r['text_exact']:>6.3f} {r['delta_text']:>+8.3f} {r['vqa_exact']:>6.3f} "
                     f"{r['chat_correct']:>6.3f} {r['recall']:>7.3f} {r['mm_score']:>9.3f}")
    text = "\n".join(lines)
    print(text)
    write_result("04_forgetting", {"base_text_exact": base_text, "runs": res}, text)


if __name__ == "__main__":
    main()
