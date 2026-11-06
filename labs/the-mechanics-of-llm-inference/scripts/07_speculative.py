"""Post section 15, speculative decoding: exactness, the E[tokens] formula, and one real draft/target pair.

Part 1 (no download): the rejection rule reproduces the target distribution exactly. Draw a million one-token
samples through the rule and compare the counts with p.
Part 2 (no download): simulate i.i.d. acceptance and compare tokens per target pass with (1 - a^(g+1)) / (1 - a).
Part 3 (needs SmolLM2-135M and SmolLM2-360M from the Hugging Face Hub): real speculative decoding with
SmolLM2-135M drafting for SmolLM2-360M. Greedy output must match plain greedy decoding token for token. Reports the
measured acceptance rate, tokens per target pass, and wall-clock speed-up, and compares the speed-up with the
post's formula using the measured acceptance rate and the measured draft cost.

    python labs/the-mechanics-of-llm-inference/scripts/07_speculative.py --out .../results/speculative_<hw>.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferlab.decoder import Decoder  # noqa: E402
from inferlab.ledger import spec_expected_tokens, spec_speedup  # noqa: E402
from inferlab.speculative import plain_generate, speculative_generate, toy_speculative_samples  # noqa: E402
from inferlab.timing import Report  # noqa: E402

PROMPTS = [
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Photosynthesis is the process by which plants",
    "Once upon a time, in a small village by the sea, there lived",
]


def part_exactness(rep: Report, n: int) -> None:
    gen = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(8, generator=gen) * 1.5, 0)
    q = torch.softmax(torch.randn(8, generator=gen) * 1.5, 0)
    x = toy_speculative_samples(p, q, n, gen)
    emp = torch.bincount(x, minlength=8).float() / n
    rep(f"Part 1. {n:,} one-token draws through the rejection rule, vocabulary of 8")
    rep(f"  total variation(draft q, target p)          = {0.5 * (q - p).abs().sum():.4f}")
    rep(f"  total variation(speculative output, target) = {0.5 * (emp - p).abs().sum():.4f}"
        f"   (sampling noise alone is about {(8 / n) ** 0.5 / 2:.4f})")
    rep(f"  acceptance rate = sum min(p, q) = {torch.minimum(p, q).sum():.3f}")


def part_formula(rep: Report, rounds: int) -> None:
    gen = torch.Generator().manual_seed(1)
    rep(f"\nPart 2. tokens per target pass, {rounds:,} simulated rounds with i.i.d. acceptance")
    rep(f"{'alpha':>6} {'gamma':>6} {'simulated':>10} {'formula':>8}")
    for alpha, gamma in [(0.5, 4), (0.8, 4), (0.8, 8), (0.95, 4)]:
        ok = torch.rand(rounds, gamma, generator=gen) < alpha
        kept = torch.where(ok.all(1), torch.full((rounds,), gamma), (~ok).float().argmax(1))
        rep(f"{alpha:>6} {gamma:>6} {(kept + 1).float().mean():>10.3f} {spec_expected_tokens(alpha, gamma):>8.3f}")
    rep(f"post's example: alpha 0.8, gamma 4 -> {spec_expected_tokens(0.8, 4):.2f} tokens per pass; "
        f"with a draft at 5% of a target step the speed-up is {spec_speedup(0.8, 4, 0.05):.2f}x")


def local_model(repo: str) -> Path | None:
    """The cached snapshot folder holding config.json and model.safetensors, or None. Never downloads."""
    from huggingface_hub import try_to_load_from_cache

    paths = [try_to_load_from_cache(repo, f) for f in ("config.json", "model.safetensors", "tokenizer.json")]
    if not all(isinstance(x, str) for x in paths):
        return None
    return Path(paths[0]).parent


def part_real(rep: Report, device: str, n_new: int) -> None:
    tdir, ddir = local_model("HuggingFaceTB/SmolLM2-360M"), local_model("HuggingFaceTB/SmolLM2-135M")
    if tdir is None or ddir is None:
        rep("\nPart 3 skipped: SmolLM2-135M and SmolLM2-360M are not in the local Hugging Face cache.")
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tdir))
    dtype = torch.float32  # fp32, so greedy speculative and greedy plain can be compared token for token
    target = Decoder.from_hf(tdir, device=device, dtype=dtype)
    draft = Decoder.from_hf(ddir, device=device, dtype=dtype)
    rep(f"\nPart 3. draft SmolLM2-135M, target SmolLM2-360M, fp32 on {device}, eager PyTorch (no CUDA graphs), "
        f"{n_new} new tokens per prompt, {len(PROMPTS)} prompts")
    prompts = [torch.tensor([tok(p)["input_ids"]], device=device) for p in PROMPTS]
    # Warm-up, so the first timed run does not pay for kernel selection.
    plain_generate(target, prompts[0], 8)
    plain_generate(draft, prompts[0], 8)
    plain_t = sum(plain_generate(target, p, n_new).seconds for p in prompts)
    draft_t = sum(plain_generate(draft, p, n_new).seconds for p in prompts)
    c = draft_t / plain_t
    rep(f"  plain decoding: target {len(prompts) * n_new / plain_t:.0f} tok/s, draft "
        f"{len(prompts) * n_new / draft_t:.0f} tok/s, so one draft step costs c = {c:.2f} target steps")
    rep(f"{'temp':>5} {'gamma':>6} {'alpha':>6} {'tok/pass':>9} {'formula':>8} {'speed-up':>9} {'formula':>8} "
        f"{'matches plain':>14}")
    for temperature in (0.0, 1.0):
        base = [plain_generate(target, p, n_new, temperature, seed=i) for i, p in enumerate(prompts)]
        for gamma in (2, 4, 8):
            runs = [speculative_generate(target, draft, p, n_new, gamma, temperature, seed=i)
                    for i, p in enumerate(prompts)]
            acc, tested = sum(r.accepted for r in runs), sum(r.tested for r in runs)
            alpha = acc / tested
            per_pass = sum(len(r.tokens) for r in runs) / sum(r.target_passes for r in runs)
            speed = sum(b.seconds for b in base) / sum(r.seconds for r in runs)
            match = "-"
            if temperature == 0:
                same = sum(r.tokens == b.tokens for r, b in zip(runs, base))
                match = f"{same}/{len(runs)} prompts"
            rep(f"{temperature:>5} {gamma:>6} {alpha:>6.3f} {per_pass:>9.2f} {spec_expected_tokens(alpha, gamma):>8.2f} "
                f"{speed:>8.2f}x {spec_speedup(alpha, gamma, c):>7.2f}x {match:>14}")
    rep("  At temperature 1 the outputs are samples, so they are not compared token by token; Part 1 checks the")
    rep("  distribution. The formula assumes the verify pass costs one plain step; for these small models it is")
    rep("  launch-bound, so that assumption holds roughly, and c is large because both models are launch-bound too.")
    sample = speculative_generate(target, draft, prompts[0], 24, 4, 0.0)
    rep("  sample (greedy, gamma 4): " + repr(PROMPTS[0][-40:] + tok.decode(sample.tokens)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--rounds", type=int, default=200_000)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = Report(args.out)
    part_exactness(rep, args.samples)
    part_formula(rep, args.rounds)
    if not args.skip_real:
        part_real(rep, "cuda" if torch.cuda.is_available() else "cpu", args.new_tokens)
    rep.save()


if __name__ == "__main__":
    main()
