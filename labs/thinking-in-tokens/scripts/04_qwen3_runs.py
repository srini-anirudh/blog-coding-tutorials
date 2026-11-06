"""Sections "Vote" and "Budgets and mode switching", measured on a real reasoning model (Qwen3-0.6B, GSM8K).

Parts (each writes results/04_<part>.md and .json):
  modes   soft /no_think and /think flags versus the hard empty-think prefill: how often is thinking off?
  vote    non-thinking mode, 16 samples per question: single-sample accuracy, vote@k, pass@k (an oracle verifier)
  budget  thinking budget 0 ... 2048 tokens with Qwen3's interrupt-and-salvage
  waits   s1 budget forcing: replace </think> with "Wait" 0, 1 or 2 times

This needs a GPU to finish in minutes. It is a 0.6B model on grade-school math, not the paper's flagship on AIME:
the shapes of the curves are the point, not the levels.

  python labs/thinking-in-tokens/scripts/04_qwen3_runs.py --part modes vote
  python labs/thinking-in-tokens/scripts/04_qwen3_runs.py --part budget waits
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from thinktok import gsm8k  # noqa: E402
from thinktok.budget import ThinkTokens, chat_prompt, think_with_budget, think_with_waits, thought_was_empty  # noqa: E402
from thinktok.util import hardware  # noqa: E402
from thinktok.vote import answer_marginal, maj_at_k, pass_at_k, plurality  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
TAG = ""


def header(cmd: str, extra: str) -> list[str]:
    import transformers

    return [f"# hardware: {hardware()}, transformers {transformers.__version__}", f"# command: {cmd}",
            f"# model: {MODEL}, {extra}", ""]


def write(name: str, lines: list[str], data: dict) -> None:
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    (LAB / "results" / f"04_{name}{TAG}.md").write_text(text)
    (LAB / "results" / f"04_{name}{TAG}.json").write_text(json.dumps(data, indent=1) + "\n")


def part_modes(model, tok, qs, cmd):
    from thinktok.hf import make_generate

    gen = make_generate(model, tok, temperature=0.6, top_p=0.95, batch_size=128)
    rows, data = [], {}
    for mode in ["think", "soft_think", "soft_no_think", "hard_no_think"]:
        prompts = [chat_prompt(tok, q["question"] + gsm8k.PROMPT_SUFFIX, mode) for q in qs]
        outs = gen(prompts, 12, {tok.convert_tokens_to_ids("<|im_end|>")})
        if mode == "hard_no_think":  # the prefill already closed the block: check the prompt, not the output
            empty = [tok.decode(p).endswith("<think>\n\n</think>\n\n") for p in prompts]
        else:
            empty = [thought_was_empty(tok.decode(o)) for o in outs]
        frac = sum(empty) / len(empty)
        data[mode] = frac
        rows.append(f"| {mode} | {frac:.1%} | {sum(empty)}/{len(empty)} |")
    lines = header(cmd, f"{len(qs)} GSM8K test questions, first 12 generated tokens, temperature 0.6") + [
        "| control | thinking off | count |", "|---|---|---|", *rows, "",
        "think = default template; soft_* append the flag to the user turn; hard_no_think = enable_thinking=False,",
        "which prefills '<think>\\n\\n</think>\\n\\n' so the model starts after the closed block.",
    ]
    write("modes", lines, data)


def part_vote(model, tok, qs, cmd, n_samples=16):
    from thinktok.hf import make_generate

    gen = make_generate(model, tok, temperature=0.7, top_p=0.8, top_k=20, batch_size=256, seed=1)
    prompts = [chat_prompt(tok, q["question"] + gsm8k.PROMPT_SUFFIX, "hard_no_think") for q in qs]
    t0 = time.perf_counter()
    outs = gen([p for p in prompts for _ in range(n_samples)], 512, {tok.convert_tokens_to_ids("<|im_end|>")})
    secs = time.perf_counter() - t0
    per_q = []
    for i, q in enumerate(qs):
        answers = [gsm8k.extract_answer(tok.decode(o, skip_special_tokens=True)) for o in outs[i * n_samples:(i + 1) * n_samples]]
        marg = answer_marginal(answers)
        per_q.append({"idx": q["idx"], "gold": q["gold"], "answers": answers, "correct": sum(a == q["gold"] for a in answers),
                      "mode": marg[0][0], "mode_share": marg[0][1]})
    ks = [1, 2, 4, 8, 16]
    vote = {k: statistics.mean(maj_at_k(p["answers"], p["gold"], k, seed=k) for p in per_q) for k in ks}
    passk = {k: statistics.mean(pass_at_k(n_samples, p["correct"], k) for p in per_q) for k in ks}
    q_right = [p["correct"] / n_samples for p in per_q]
    right_is_mode = [p for p in per_q if p["mode"] == p["gold"]]
    minority_recovered = [p for p in right_is_mode if p["correct"] / n_samples <= 0.5]
    confident_wrong = [p for p in per_q if p["mode"] != p["gold"] and p["mode_share"] >= 0.5]
    lines = header(cmd, f"{len(qs)} GSM8K test questions, non-thinking mode (hard prefill), {n_samples} samples each, "
                        f"temperature 0.7, top-p 0.8, top-k 20, max 512 tokens; generation took {secs:.0f} s") + [
        "| k samples | vote@k (plurality) | pass@k (oracle verifier picks) |", "|---|---|---|",
        *[f"| {k} | {vote[k]:.3f} | {passk[k]:.3f} |" for k in ks], "",
        f"mean single-sample accuracy: {statistics.mean(q_right):.3f}",
        f"questions where the right answer is the mode of the {n_samples} samples: {len(right_is_mode)}/{len(qs)}",
        f"  of those, right in at most half the samples (a minority the vote still recovers): {len(minority_recovered)}",
        f"questions where a wrong answer is the mode with at least half the samples (vote confirms an error): {len(confident_wrong)}",
        "",
        f"vote@k for k < {n_samples} averages 64 random size-k subsets of the {n_samples} samples; pass@k is the unbiased estimator.",
    ]
    write("vote", lines, {"vote_at_k": vote, "pass_at_k": passk, "seconds": secs, "per_question": per_q})


def _score(tok, qs, traces):
    accs, think, trunc = [], [], []
    for q, t in zip(qs, traces):
        accs.append(gsm8k.extract_answer(tok.decode(t.answer, skip_special_tokens=True)) == q["gold"])
        think.append(len(t.thinking))
        trunc.append(t.truncated)
    return statistics.mean(accs), statistics.mean(think), statistics.mean(trunc)


def _toks(tok):
    return ThinkTokens(think_end=tok.convert_tokens_to_ids("</think>"), eos=tok.convert_tokens_to_ids("<|im_end|>"))


def part_budget(model, tok, qs, cmd, budgets=(0, 64, 128, 256, 512, 1024, 2048)):
    from thinktok.hf import make_generate

    prompts = [chat_prompt(tok, q["question"] + gsm8k.PROMPT_SUFFIX, "think") for q in qs]
    rows, data = [], {}
    for b in budgets:
        gen = make_generate(model, tok, temperature=0.6, top_p=0.95, batch_size=128, seed=2)
        t0 = time.perf_counter()
        traces = think_with_budget(gen, prompts, tok, _toks(tok), b, max_answer=512)
        acc, think, trunc = _score(tok, qs, traces)
        data[b] = {"acc": acc, "mean_thinking_tokens": think, "truncated": trunc, "seconds": time.perf_counter() - t0}
        rows.append(f"| {b} | {acc:.3f} | {think:.0f} | {trunc:.0%} | {data[b]['seconds']:.0f} |")
        print(rows[-1], flush=True)
    example = next((t for t in traces if t.truncated), None)
    lines = header(cmd, f"{len(qs)} GSM8K test questions, thinking mode, temperature 0.6, top-p 0.95, top-k 20, "
                        "answer phase up to 512 tokens") + [
        "| thinking budget | accuracy | mean thinking tokens used | cut off by the budget | seconds |", "|---|---|---|---|---|",
        *rows, "",
        "Budget 0 is the hard switch (empty think block prefilled). At the cap the controller splices in Qwen3's",
        "stop-thinking sentence and </think>, then the model answers from the partial trace.",
    ]
    write("budget", lines, data)


def part_waits(model, tok, qs, cmd, cap=2048, waits=(0, 1, 2)):
    from thinktok.hf import make_generate

    prompts = [chat_prompt(tok, q["question"] + gsm8k.PROMPT_SUFFIX, "think") for q in qs]
    rows, data = [], {}
    for w in waits:
        gen = make_generate(model, tok, temperature=0.6, top_p=0.95, batch_size=128, seed=3)
        t0 = time.perf_counter()
        traces = think_with_waits(gen, prompts, tok, _toks(tok), w, cap=cap, max_answer=512)
        acc, think, trunc = _score(tok, qs, traces)
        used = statistics.mean(t.waits for t in traces)
        changed = None
        data[w] = {"acc": acc, "mean_thinking_tokens": think, "hit_cap": trunc, "mean_waits_used": used,
                   "seconds": time.perf_counter() - t0,
                   "answers": [gsm8k.extract_answer(tok.decode(t.answer, skip_special_tokens=True)) for t in traces]}
        if w and 0 in data:
            base = [a == q["gold"] for a, q in zip(data[0]["answers"], qs)]
            now = [a == q["gold"] for a, q in zip(data[w]["answers"], qs)]
            changed = (sum(n and not b for b, n in zip(base, now)), sum(b and not n for b, n in zip(base, now)))
            data[w]["fixed_vs_broken"] = changed
        rows.append(f"| {w} | {acc:.3f} | {think:.0f} | {used:.2f} | {trunc:.0%} | "
                    f"{'-' if changed is None else f'{changed[0]} / {changed[1]}'} |")
        print(rows[-1], flush=True)
    lines = header(cmd, f"{len(qs)} GSM8K test questions, thinking mode, cap {cap} thinking tokens, temperature 0.6") + [
        "| Waits forced | accuracy | mean thinking tokens | Waits used | hit the cap | right now, wrong at 0 / wrong now, right at 0 |",
        "|---|---|---|---|---|---|", *rows, "",
        "Each run samples afresh, so some flips between rows are sampling noise rather than the Wait.",
    ]
    write("waits", lines, data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["modes", "vote", "budget", "waits"])
    ap.add_argument("--n", type=int, default=100, help="questions for modes, budget and waits")
    ap.add_argument("--n-vote", type=int, default=200, help="questions for the vote part")
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--cap", type=int, default=2048, help="thinking cap for the waits part")
    ap.add_argument("--samples", type=int, default=16, help="samples per question for the vote part")
    ap.add_argument("--tag", default="", help="suffix for result files (smoke runs)")
    args = ap.parse_args()
    global TAG
    TAG = args.tag
    cmd = "python labs/thinking-in-tokens/scripts/04_qwen3_runs.py " + " ".join(sys.argv[1:])
    from thinktok.hf import load

    model, tok = load(MODEL)
    qs = gsm8k.load_questions(max(args.n, args.n_vote), seed=0)
    for part in args.part:
        t0 = time.perf_counter()
        sub = qs[: args.n_vote] if part == "vote" else qs[: args.n]
        if part == "modes":
            part_modes(model, tok, sub, cmd)
        elif part == "vote":
            part_vote(model, tok, sub, cmd, n_samples=args.samples)
        elif part == "budget":
            part_budget(model, tok, sub, cmd, budgets=args.budgets)
        elif part == "waits":
            part_waits(model, tok, sub, cmd, cap=args.cap)
        print(f"[{part} done in {time.perf_counter() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
