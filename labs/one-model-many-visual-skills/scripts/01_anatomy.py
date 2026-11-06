"""Anatomy of an instruction example, and why "50% of examples" is not "50% of the signal".

1. Serialize a two-turn, two-image conversation, expand the image placeholders, print the loss mask.
2. Count positions and loss tokens at this lab's size (9 visual tokens per image) and at LLaVA-1.5's
   size (576 per image). The second is arithmetic on the same template, not a run of LLaVA.
3. The post's normalization example (VQA answers of 2 tokens, captions of 150, half each) under
   per-token and per-example averaging, then the same measured on a real batch from this lab's data.
4. Shares in examples, compute and loss tokens for a made-up collection, before and after a per-source cap.

    python labs/one-model-many-visual-skills/scripts/01_anatomy.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillmix.chat import VOCAB, count_positions, serialize  # noqa: E402
from skillmix.mixture import Source, cap, shares, signal_share  # noqa: E402
from skillmix.model import N_VISUAL  # noqa: E402
from skillmix.runs import write_result  # noqa: E402
from skillmix.train import RunConfig, collate, sample_batch, token_weights  # noqa: E402

TURNS = [("user", "<image> <image> what color is the cup ?"), ("assistant", "the cup is red ."),
         ("user", "where is the ball ?"), ("assistant", "the ball is at the top left .")]


def show(s) -> str:
    rows, i = [], 0
    while i < len(s.ids):
        if s.kinds[i] == "visual":
            j = i
            while j < len(s.ids) and s.kinds[j] == "visual":
                j += 1
            rows.append(f"  [{i:3d}-{j - 1:3d}] {'<visual> x ' + str(j - i):<18} visual     mask 0")
            i = j
            continue
        rows.append(f"  [{i:3d}]     {VOCAB.itos[s.ids[i]]:<18} {s.kinds[i]:<10} mask {s.mask[i]}")
        i += 1
    return "\n".join(rows)


def main() -> None:
    out = []
    small = serialize(TURNS, N_VISUAL)
    out.append("1. One two-turn, two-image example as the model sees it (9 visual tokens per image):")
    out.append(show(small))
    c_small = count_positions(small)
    c_llava = count_positions(serialize(TURNS, 576))
    out.append("\n2. Positions and loss tokens")
    for name, c in (("this lab, 9 per image", c_small), ("LLaVA-1.5 size, 576 per image (arithmetic)", c_llava)):
        out.append(f"  {name}: {c['total']} positions, {c['visual']} visual, {c['loss_tokens']} carry loss "
                   f"({100 * c['loss_tokens'] / c['total']:.1f}%)")

    post = signal_share({"vqa": 0.5, "caption": 0.5}, {"vqa": 2, "caption": 150}, "token")
    post_ex = signal_share({"vqa": 0.5, "caption": 0.5}, {"vqa": 2, "caption": 150}, "example")
    out.append("\n3. Share of the loss signal that the VQA half receives (post's numbers: 2 vs 150 loss tokens)")
    out.append(f"  per-token average:   {100 * post['vqa']:.1f}%")
    out.append(f"  per-example average: {100 * post_ex['vqa']:.1f}%")
    rng = random.Random(0)
    measured = {}
    for norm in ("token", "example"):
        tot, n_loss = 0.0, {"vqa": [], "caption": []}
        for _ in range(20):
            b = collate(sample_batch(RunConfig(mixture={"vqa": 0.5, "caption": 0.5}, batch=64), rng))
            w = token_weights(b["mask"], norm)
            tot += sum(float(wi) for wi, s in zip(w, b["sources"]) if s == "vqa") / 20
            for s, m in zip(b["sources"], b["mask"]):
                n_loss[s].append(int(m.sum()))
        measured[norm] = tot
    mean_loss = {k: sum(v) / len(v) for k, v in n_loss.items()}
    out.append(f"  measured on 20 batches of this lab's data (VQA answers {mean_loss['vqa']:.1f} loss tokens, "
               f"captions {mean_loss['caption']:.1f}):")
    out.append(f"    per-token {100 * measured['token']:.1f}%, per-example {100 * measured['example']:.1f}%")

    # A made-up collection, chosen only to show that the three shares disagree. Not real dataset sizes.
    coll = [Source("short VQA", 800_000, 576, 20, 2), Source("detailed captions", 100_000, 576, 160, 150),
            Source("documents, 5 tiles", 150_000, 2880, 40, 20), Source("text-only", 100_000, 0, 300, 250)]
    out.append("\n4. A made-up collection (illustrative sizes): share of examples / compute / loss tokens")
    for title, c in (("before capping", coll), ("after a 300K per-source cap", cap(coll, 300_000))):
        out.append(f"  {title}:")
        for name, s in shares(c).items():
            out.append(f"    {name:<20} {100 * s['examples']:5.1f}% / {100 * s['compute']:5.1f}% / "
                       f"{100 * s['loss_tokens']:5.1f}%")
    text = "\n".join(out)
    print(text)
    write_result("01_anatomy", {"lab_counts": c_small, "llava_size_counts": c_llava,
                                "post_example_vqa_share": {"token": post["vqa"], "example": post_ex["vqa"]},
                                "measured_vqa_share": measured, "mean_loss_tokens": mean_loss}, text, sweeping=False)


if __name__ == "__main__":
    main()
