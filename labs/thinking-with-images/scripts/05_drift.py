"""Long generations look less: attention share on image tokens along a long trace, and re-injecting a crop.

The model is made to write 1024 tokens about the photo (end-of-text is suppressed, so late text may ramble) at 896 px (768 visual tokens). One teacher-forced pass then gives,
for each generated position, the share of attention on image tokens, averaged over all heads and layers. A second pass
inserts the plaque crop (192 tokens) after generated token 256 and measures the share again. Needs eager attention.
About 10 min on one H100.
    python labs/thinking-with-images/scripts/05_drift.py --n 12 --gen-tokens 1024
"""
import torch
from _common import command, out_path, parser

from lookagain.drift import binned, drift_one
from lookagain.scene import make_scene
from lookagain.util import write

BIN = 128


def main():
    ap = parser(__doc__, 12)
    ap.add_argument("--gen-tokens", type=int, default=1024)
    args = ap.parse_args()
    if args.stub:
        raise SystemExit("05_drift.py reads real attention maps; there is no stub mode.")
    from lookagain.vlm import QwenVLM
    vlm = QwenVLM(attn="eager")
    runs = [drift_one(vlm, make_scene(4000 + i), gen_tokens=args.gen_tokens, reinject_at=256) for i in range(args.n)]
    L = min(len(r["base"]) for r in runs)
    L -= L % BIN
    base = torch.stack([r["base"][:L] for r in runs]).mean(0)
    re = torch.stack([r["reinjected"][:L] for r in runs]).mean(0)
    uni = torch.stack([r["uniform"][:L] for r in runs]).mean(0)
    k = runs[0]["k"]
    rows = [f"n={args.n} scenes; image at 896 x 672 = {runs[0]['n_image_tokens']} visual tokens; prompt "
            f"{runs[0]['prompt_len']} tokens; {L} generated tokens each",
            f"crop of the plaque cell (192 tokens) re-injected after generated token {k}", "",
            f"{'tokens':>10} {'share on image':>15} {'vs first bin':>13} {'with re-injection':>18} {'image / context':>16}"]
    b0 = binned(base, BIN)[0]
    for i, (b, r, u) in enumerate(zip(binned(base, BIN), binned(re, BIN), binned(uni, BIN))):
        rows.append(f"{i * BIN:>5}-{(i + 1) * BIN - 1:<4} {b:>15.3f} {b / b0:>13.2f} {r:>18.3f} {u:>16.3f}")
    rows += ["", "share on image: attention mass on image tokens, mean over heads, layers and scenes.",
             "image / context: N_v / context length, the share the image would get if attention were spread evenly.",
             "The measured share is well below image / context. This script does not separate out attention-sink tokens.",
             "Before the re-injection point the two passes see identical inputs, so their columns agree.", "",
             "First trace (scene 4000), first 600 characters:", runs[0]["trace"][:600]]
    write(out_path(args, "05_drift"), command("05_drift.py"), rows,
          {"bin": BIN, "base": binned(base, BIN), "reinjected": binned(re, BIN), "uniform": binned(uni, BIN),
           "reinject_at": k, "generated_tokens_used": L})


if __name__ == "__main__":
    main()
